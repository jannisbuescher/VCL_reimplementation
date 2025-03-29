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

from model import VariationalMLP
from loss import variational_loss
from mnist_perm import create_data_loaders, convert_to_jax

class TrainState(train_state.TrainState):
    """Training state for the variational model."""
    prior_params: Dict
    rng: jax.random.PRNGKey

def create_train_state(
    model: VariationalMLP,
    learning_rate: float = 0.001,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> TrainState:
    """Create initial training state."""
    params = model.init(rng, jnp.ones((1, 784)))['params']
    prior_params = params  # Initialize prior with same parameters
    
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        prior_params=prior_params,
        tx=tx,
        rng=rng
    )

@partial(jax.jit, static_argnums=(2,))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    num_samples: int = 10  # Samples for Monte Carlo estimation
) -> Tuple[TrainState, Dict]:
    """Single training step with Monte Carlo estimation."""
    images, labels = batch
    
    def loss_fn(params):
        def sample_step(i, state_tuple):
            logits_samples, metrics_samples, total_loss = state_tuple
            
            # Generate new RNG key for each sample
            rng = jax.random.PRNGKey(i)  # Use simple key instead of fold_in
            logits = state.apply_fn({'params': params}, images, rng)
            
            # Compute loss for each sample
            loss, metrics = variational_loss(
                params=params,
                prior_params=state.prior_params,
                logits=logits,
                labels=labels,
            )
            
            # Update samples
            logits_samples = logits_samples.at[i].set(logits)
            metrics_samples = {k: metrics_samples[k].at[i].set(metrics[k]) 
                             for k in metrics_samples.keys()}
            
            return (logits_samples, metrics_samples, total_loss + loss)
        
        # Initialize arrays for samples
        init_state = (
            jnp.zeros((num_samples, images.shape[0], 10)),  # logits_samples
            {  # metrics_samples
                'nll': jnp.zeros(num_samples),
                'kl_div': jnp.zeros(num_samples),
                'total_loss': jnp.zeros(num_samples)
            },
            0  # total_loss
        )
        
        # Run Monte Carlo sampling
        logits_samples, metrics_samples, total_loss = jax.lax.fori_loop(
            lower=0,
            upper=num_samples,
            body_fun=sample_step,
            init_val=init_state
        )


        # Average metrics across samples
        avg_metrics = {
            k: jnp.mean(metrics_samples[k])
            for k in metrics_samples.keys()
        }
        
        return total_loss, avg_metrics
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    return state, metrics

@partial(jax.jit, static_argnums=(2,))
def evaluate(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    num_samples: int = 10
) -> Dict:
    """Evaluate the model on test data."""
    metrics_list = []
    correct = 0
    total = 0
    
    images, labels = batch

    def logit_sample_step(state, i):
        rng = jax.random.PRNGKey(i)
        logits = state.apply_fn({'params': state.params}, images, rng)
        return (state, logits)
    
    # _, logits_samples = jax.lax.scan(logit_sample_step, state, jnp.arange(1), length=1)
    _, logits_samples = logit_sample_step(state, 0)
    logits_samples = jnp.expand_dims(logits_samples, axis=0)

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

def train_continual(
    model: VariationalMLP,
    num_tasks: int = 5,
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    num_samples: int = 10,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
):
    """Train the model on multiple tasks sequentially."""
    # Create data loaders for all tasks
    data_loaders = create_data_loaders(
        batch_size=batch_size,
        num_tasks=num_tasks,
    )
    
    # Initialize training state
    state = create_train_state(model, learning_rate, rng)
    
    # Train on each task
    for task_id, (train_loader, test_loader) in enumerate(data_loaders):
        print(f"\nTraining on Task {task_id + 1}/{num_tasks}")
        
        for epoch in range(num_epochs):
            

            # Evaluation loop
            metrics_list = []
            # Evaluate on test set
            for batch in test_loader:

                # if len(metrics_list) == 0:
                #     batch1 = convert_to_jax(batch)
                #     # get random number

                #     debug_predict(state, batch1[0][3], batch1[1][3])

                metrics_eval, correct, total = evaluate(state, convert_to_jax(batch), num_samples)
                metrics_list.append(metrics_eval)

                # Average metrics
            avg_metrics = {
                k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            avg_metrics['accuracy'] = correct / total
            eval_test_acc = avg_metrics['accuracy']
            eval_test_loss = avg_metrics['total_loss']
            del avg_metrics

            metrics_list_train = []
            # Evaluate on test set
            for batch in train_loader:
                metrics_eval, correct, total = evaluate(state, convert_to_jax(batch), num_samples)
                metrics_list_train.append(metrics_eval)

                # Average metrics
            avg_metrics_train = {
                k: jnp.mean(jnp.array([m[k] for m in metrics_list_train]))
                for k in metrics_list_train[0].keys()
            }
            avg_metrics_train['accuracy'] = correct / total
            eval_loss_train = avg_metrics_train['total_loss']
            eval_train_acc = avg_metrics_train['accuracy']
            del avg_metrics_train



            # Training loop
            metrics_list = []
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                batch = convert_to_jax(batch)
                state, metrics = train_step(state, batch, num_samples)
                metrics_list.append(metrics)
            
            # Average training metrics
            avg_metrics = {
                k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            train_loss_train = avg_metrics['total_loss']
            del avg_metrics



            # Print metrics every x epochs
            if (epoch + 1) % 1 == 0:
                print(f"\nEpoch {epoch + 1}")
                print(f"Train Loss: {train_loss_train:.4f}")
                print(f"Train Loss eval: {eval_loss_train:.4f}")
                print(f"Test Loss eval: {eval_test_loss:.4f}")
                print(f"Train Accuracy: {eval_train_acc:.4f}")
                print(f"Test Accuracy: {eval_test_acc:.4f}")
        
        # Update prior parameters with current posterior
        state = state.replace(prior_params=state.params)

    for task_id, (_, test_loader) in enumerate(data_loaders):
        print(f"\nTesting on Task {task_id + 1}/{num_tasks}")
        for batch in test_loader:
            metrics_eval, correct, total = evaluate(state, convert_to_jax(batch), num_samples)
            metrics_list.append(metrics_eval)
            
            # Average metrics
            avg_metrics = {
                k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            avg_metrics['accuracy'] = correct / total
            eval_test_acc = avg_metrics['accuracy']
        print(f"Test Accuracy: {eval_test_acc:.4f}")
            
    
    return state

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
        logits = state.apply_fn({'params': state.params}, image[None], rng)
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

if __name__ == "__main__":
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(123)
    np.random.seed(123)
    
    # Create and train model
    model = VariationalMLP()
    state = train_continual(
        model=model,
        num_tasks=5,
        num_epochs=2,
        batch_size=128,
        learning_rate=0.001,
        num_samples=10,
        rng=rng
    )

    