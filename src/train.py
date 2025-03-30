import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
import multiprocessing
import os
from functools import partial

# Set multiprocessing start method to 'spawn' to avoid deadlocks
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['PYTHONUNBUFFERED'] = '1'

from .model import VariationalMLP
from .loss import variational_loss
from .mnist_perm import create_data_loaders as get_dataloaders_mnist_perm
from .mnist_split import get_dataloaders as get_dataloaders_mnist_split
from . import utils

# CONSTANTS
DEBUG = True

if DEBUG:
    PRINT_EVERY = 1
else:
    PRINT_EVERY = 20



class TrainState(train_state.TrainState):
    """Training state for the variational model."""
    prior_params: Dict
    rng: jax.random.PRNGKey
    curr_head: int = 0

def create_train_state(
    model: VariationalMLP,
    learning_rate: float = 0.001,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> TrainState:
    """Create initial training state."""
    params = model.init(rng, jnp.ones((1, 784)))['params']
    prior_params = params  # Initialize prior with same parameters
    
    tx = optax.adam(learning_rate)
    # apply_fn = jax.jit(model.apply, static_argnums=(2,))
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        prior_params=prior_params,
        tx=tx,
        rng=rng
    )

def copy_train_state(train_state: TrainState, lr: float = 0.0005) -> TrainState:
    return TrainState.create(
        apply_fn=train_state.apply_fn,
        params=train_state.params,
        prior_params=train_state.prior_params,
        tx=optax.adam(lr),
        rng=train_state.rng,
        curr_head=train_state.curr_head
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.random.PRNGKey,
    num_samples: int = 10, # Samples for Monte Carlo estimation,
) -> Tuple[TrainState, Dict]:
    """Single training step with Monte Carlo estimation."""
    images, labels = batch
    
    def loss_fn(params, rng):        
        def sample_step(rng, state):
            logits = state.apply_fn({'params': params}, images, rng, state.curr_head)
            loss, metrics = variational_loss(
                params=params,
                prior_params=state.prior_params,
                logits=logits,
                labels=labels,
            )            
            return (logits, metrics, loss)
        
        # Monte Carlo sampling
        rngs = jax.random.split(rng, num_samples+1)
        rng = rngs[0]
        subrng = rngs[1:]
        _, metrics, loss = jax.vmap(sample_step, in_axes=(0, None))(subrng, state)
        
        return jnp.mean(loss), (metrics, rng)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (metrics, rng)), grads = grad_fn(state.params, rng)
    
    state = state.apply_gradients(grads=grads)
    return state, metrics

@partial(jax.jit, static_argnums=(3,))
def evaluate(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.random.PRNGKey,
    num_samples: int = 10
) -> Dict:
    """Evaluate the model on test data."""
    metrics_list = []
    correct = 0
    total = 0
    
    images, labels = batch

    def logit_sample_step(rng, state):
        logits = state.apply_fn({'params': state.params}, images, rng, state.curr_head)
        return logits
    
    subkeys = jax.random.split(rng, num_samples + 1)
    rng = subkeys[0]
    subkeys = subkeys[1:]
    logits_samples = jax.vmap(logit_sample_step, in_axes=(0,None))(subkeys, state)

    # jax.debug.callback(lambda x: print(f"Logits: {x[0, :2, :]}"), logits_samples)

    # Average predictions
    logits = jnp.mean(logits_samples, axis=0)
    predictions = jnp.argmax(logits, axis=-1)
    
    # Compute accuracy
    correct += jnp.sum(predictions == labels)
    total += len(labels)
    
    # Compute loss
    _, metrics = variational_loss(
        params=state.params,
        prior_params=state.prior_params,
        logits=logits,
        labels=labels
    )
    metrics_list.append(metrics)  
    return metrics, correct, total


def eval_task(dataloader, state, rng, num_samples, verbose=False):
    big_correct = 0
    big_total = 0

    for batch in dataloader:
        rng, subrng = jax.random.split(rng)
        _, correct, total = evaluate(state, utils.convert_to_jax(batch), subrng, num_samples)
        big_correct += correct
        big_total += total
        
    eval_test_acc = big_correct / big_total
    if verbose:
        print(f"Test Accuracy: {eval_test_acc:.4f}")
    return rng, eval_test_acc

def train_continual(
    model: VariationalMLP,
    task: str,
    num_tasks: int = 5,
    num_epochs: int = 10,
    num_coreset_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    num_samples: int = 10,
    use_coreset: bool = False,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
):
    """Train the model on multiple tasks sequentially."""
    # Create data loaders for all tasks
    data_loaders, nb_heads = get_dataloaders(
        task,
        batch_size=batch_size,
        num_tasks=num_tasks,
    )

    if use_coreset:
        coreset_size = 200
        coreset_loaders = get_coreset_loader(batch_size, data_loaders, coreset_size)


    model.num_heads = nb_heads
    multi_head = nb_heads > 1

    avg_accuracies = []
    
    # Initialize training state
    rng, subrng = jax.random.split(rng)
    state = create_train_state(model, learning_rate, subrng)
    
    # Train on each task
    for task_id, (train_loader, test_loader) in enumerate(data_loaders):
        print(f"\nTraining on Task {task_id + 1}/{num_tasks}")

        if multi_head:
            state = state.replace(curr_head=task_id)
        
        for epoch in range(num_epochs):
            
            # Training loop
            metrics_list = []
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                batch = utils.convert_to_jax(batch)

                rng, subrng = jax.random.split(rng)
                state, metrics = train_step(state, batch, subrng, num_samples)
                metrics_list.append(metrics)
            
            # Average training metrics
            avg_metrics = {
                k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            train_loss_train = avg_metrics['total_loss']
            del avg_metrics


            # Print metrics every x epochs
            if (epoch + 1) % PRINT_EVERY == 0:

                # Evaluation loop Test set
                metrics_list = []
                big_correct = 0
                big_total = 0
                # Evaluate on test set
                for batch in test_loader:
                    
                    rng, subrng = jax.random.split(rng)
                    metrics_eval, correct, total = evaluate(state, utils.convert_to_jax(batch), subrng, num_samples)
                    metrics_list.append(metrics_eval)
                    big_correct += correct
                    big_total += total

                    # Average metrics
                avg_metrics = {
                    k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
                    for k in metrics_list[0].keys()
                }
                eval_test_acc = big_correct / big_total
                eval_test_loss = avg_metrics['total_loss']
                del avg_metrics, big_correct, big_total


                # Evaluation loop Train set
                metrics_list_train = []
                big_correct = 0
                big_total = 0
                # Evaluate on test set
                for batch in train_loader:
                    rng, subrng = jax.random.split(rng)
                    metrics_eval, correct, total = evaluate(state, utils.convert_to_jax(batch), subrng, num_samples)
                    metrics_list_train.append(metrics_eval)
                    big_correct += correct
                    big_total += total

                    # Average metrics
                avg_metrics_train = {
                    k: jnp.mean(jnp.array([m[k] for m in metrics_list_train]))
                    for k in metrics_list_train[0].keys()
                }
                eval_loss_train =  avg_metrics_train['total_loss']
                eval_train_acc = big_correct / big_total
                del avg_metrics_train, big_correct, big_total


                print(f"\nEpoch {epoch + 1}")
                print(f"Train Loss: {train_loss_train:.4f}")
                print(f"Train Loss eval: {eval_loss_train:.4f}")
                print(f"Test Loss eval: {eval_test_loss:.4f}")
                print(f"Train Accuracy: {eval_train_acc:.4f}")
                print(f"Test Accuracy: {eval_test_acc:.4f}")
        
        # Update prior parameters with current posterior
        state = state.replace(prior_params=state.params)


        # train on coreset
        rng, subrng = jax.random.split(rng)
        if use_coreset:
            new_train_state = copy_train_state(state)
            for epoch in range(num_coreset_epochs):
                for batch in coreset_loaders[task_id]:
                    batch = utils.convert_to_jax(batch)
                    rng, subrng = jax.random.split(rng)
                    new_train_state, metrics = train_step(new_train_state, batch, subrng, num_samples)
                    metrics_list.append(metrics)
            
            for eval_task_id, (_, test_loader) in enumerate(data_loaders[:task_id+1]):
                if multi_head:
                    new_train_state = new_train_state.replace(curr_head=eval_task_id)
                print(f"\nTesting on Task with upgraded model trained on coreset {eval_task_id + 1}/{task_id+1}")
                rng, acc = eval_task(test_loader, new_train_state, rng, num_samples, verbose=True)
                

        
        avg_acc = 0
        for eval_task_id, (_, test_loader) in enumerate(data_loaders[:task_id+1]):
            if multi_head:
                state = state.replace(curr_head=eval_task_id)
            print(f"\nTesting on Task {eval_task_id + 1}/{task_id+1}")
            rng, acc = eval_task(test_loader, state, rng, num_samples, verbose=True)
            avg_acc += acc
        avg_acc /= task_id+1
        avg_accuracies.append(avg_acc)

    for task_id, (_, test_loader) in enumerate(data_loaders):
        if multi_head:
            state = state.replace(curr_head=task_id)
        print(f"\nTesting on Task {task_id + 1}/{num_tasks}")
        eval_task(test_loader, state, rng, num_samples, verbose=True)
            
    
    return state, avg_accuracies

def debug_predict(
    state: TrainState,
    image: jnp.ndarray,
    label: jnp.ndarray,
    num_samples: int = 1
) -> None:
    """
    Debug prediction for a single datapoint.
    Prints all intermediate values and probabilities.
    """
    print("\n=== Debug Prediction ===")
    print(f"Input shape: {image.shape}")
    print(f"True label: {label}")
    
    # Get predictions from multiple samples
    logits_samples = []
    for i in range(num_samples):
        rng = jax.random.PRNGKey(i)
        logits = state.apply_fn({'params': state.params}, image[None], rng, state.curr_head)
        logits_samples.append(logits)
        print(f"\nSample {i+1}:")
        print(f"Logits: {logits[0]}")
        print(f"Softmax probabilities: {jax.nn.softmax(logits[0])}")
    
    # Average predictions
    avg_logits = jnp.mean(jnp.stack(logits_samples), axis=0)
    avg_probs = jax.nn.softmax(avg_logits)
    prediction = jnp.argmax(avg_logits)
    
    print("\n=== Final Results ===")
    print(f"Average logits: {avg_logits[0]}")
    print(f"Average probabilities: {avg_probs[0]}")
    print(f"Predicted label: {prediction}")
    print(f"Correct: {prediction == label}")
    
    # Print layer-wise information
    print("\n=== Layer-wise Information ===")
    for layer_name, params in state.params.items():
        if 'Dense' in layer_name:
            print(f"\nLayer: {layer_name}")
            print(f"Weights mu shape: {params['weights_mu'].shape}")
            print(f"Weights var shape: {params['weights_var'].shape}")
            print(f"Bias mu shape: {params['bias_mu'].shape}")
            print(f"Bias var shape: {params['bias_var'].shape}")


def get_dataloaders(
    task: str,
    batch_size: int = 128,
    num_tasks: int = 5
):
    if task == 'mnist_perm':
        return get_dataloaders_mnist_perm(batch_size, num_tasks), 1
    elif task == 'mnist_split':
        return get_dataloaders_mnist_split(batch_size, num_tasks), num_tasks
    
def get_coreset_loader(
    batch_size: int,
    data_loaders,
    coreset_size: int = 200
):
    return utils.get_coreset_dataloader(data_loaders, batch_size, coreset_size)


if __name__ == "__main__":
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(123)
    np.random.seed(123)

    model = VariationalMLP()
    # Create and train model
    state, avg_accuracies = train_continual(
        model=model,
        task='mnist_perm',
        num_tasks=3,
        num_epochs=2,
        num_coreset_epochs=1,
        batch_size=256,
        learning_rate=0.001,
        num_samples=10,
        use_coreset=True,
        rng=rng
    )

    print(f"Average accuracies: {avg_accuracies}")