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
from torch.utils.data import DataLoader

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['PYTHONUNBUFFERED'] = '1'

from model_head_minimum_working_version import VariationalMLP
from loss_mwv_plus import variational_loss
from mnist_perm import create_data_loaders as get_dataloaders_mnist_perm
from mnist_split import get_dataloaders as get_dataloaders_mnist_split
import utils

# CONSTANTS
DEBUG = __name__ == "__main__"

if DEBUG:
    PRINT_EVERY = 1
else:
    PRINT_EVERY = 20



class TrainState(train_state.TrainState):
    prior_params: Dict

def create_train_state(
    model: VariationalMLP,
    learning_rate: float = 0.001,
) -> TrainState:
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 784)))['params']

    prior_params = jax.tree.map(lambda x: x.copy(), params)
      
    for key in prior_params.keys():
        for subkey in prior_params[key].keys():
            if 'var' in subkey:
                prior_params[key][subkey] = jnp.zeros_like(prior_params[key][subkey])
            elif 'mu' in subkey:
                prior_params[key][subkey] = jnp.zeros_like(prior_params[key][subkey])
            
    
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        prior_params=prior_params,
        tx=tx
    )

def train_state_replace(state: TrainState) -> TrainState:
    prior_params = jax.tree.map(lambda x: x.copy(), state.params)
    state = state.replace(prior_params=prior_params)
    return state

def train_state_copy(state: TrainState) -> TrainState:
    params = jax.tree.map(lambda x: x.copy(), state.params)
    prior_params = jax.tree.map(lambda x: x.copy(), state.params)
    return TrainState.create(
        apply_fn=state.apply_fn,
        params=params,
        prior_params=prior_params,
        tx=state.tx
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.random.PRNGKey,
    head_id: int = 0,
    kl_weight: float = 1e-6
) -> Tuple[TrainState, Dict]:
    images, labels = batch
    
    def loss_fn(params, rng):        
        def sample_step(rng, state):
            # jax.debug.callback(lambda x: print(x), images)
            logits = state.apply_fn({'params': params}, images, rng, head_id=head_id)
            # jax.debug.callback(lambda x: print(x), logits)
            loss, metrics = variational_loss(
                params=params,
                prior_params=state.prior_params,
                logits=logits,
                labels=labels,
                head_id=head_id,
                kl_weight=kl_weight
            )
            return (logits, metrics, loss)
        
        rng, subrng = jax.random.split(rng)
        _, metrics, loss = sample_step(subrng, state)
        
        return loss, (metrics, rng)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (metrics, rng)), grads = grad_fn(state.params, rng)
    
    state = state.apply_gradients(grads=grads)
    return state, (metrics, rng)

@partial(jax.jit, static_argnums=(3,))
def evaluate(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.random.PRNGKey,
    head_id: int = 0
) -> Tuple[int, int, jax.random.PRNGKey]:
    correct = 0
    total = 0
    
    images, labels = batch

    def logit_sample_step(rng, state):
        logits = state.apply_fn({'params': state.params}, images, rng, head_id=head_id)
        return logits
    
    rng, subkey = jax.random.split(rng)
    logits = logit_sample_step(subkey, state)

    predictions = jnp.argmax(logits, axis=-1)
    
    correct += jnp.sum(predictions == labels)
    total += len(labels)
    
    return correct, total, rng


def eval_task(dataloader, state, rng, head_id: int = 0, verbose=False):
    big_correct = 0
    big_total = 0

    for batch in dataloader:
        rng, subrng = jax.random.split(rng)
        correct, total, rng = evaluate(state, utils.convert_to_jax(batch), subrng, head_id=head_id)
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
    batch_size: int = 128,
    learning_rate: float = 0.001,
    use_coreset: bool = False,
    num_coreset_epochs: int = 10,
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
        coreset_loaders = get_coreset_loader(batch_size, data_loaders)

    model.num_heads = nb_heads

    use_multiple_heads = model.num_heads > 1

    avg_accuracies = []
    
    state = create_train_state(model, learning_rate)
    
    for task_id, (train_loader, test_loader) in enumerate(data_loaders):
        print(f"\nTraining on Task {task_id + 1}/{num_tasks}")

        if use_multiple_heads:
            head_id = task_id
        else:
            head_id = 0
        
        for epoch in range(num_epochs):
            metrics_list = []
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                batch = utils.convert_to_jax(batch)

                rng, subrng = jax.random.split(rng)
                state, (metrics, rng) = train_step(state, batch, subrng, head_id=head_id, kl_weight=1/60000)
                metrics_list.append(metrics)
            
            # Average training metrics
            avg_metrics = {
                k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            train_loss_train = avg_metrics['total_loss']
            train_nll_train = avg_metrics['nll']
            train_kl_train = avg_metrics['kl_div']
            del avg_metrics


            # Printing
            if (epoch + 1) % PRINT_EVERY == 0:
                big_correct = 0
                big_total = 0
                for batch in test_loader:
                    rng, subrng = jax.random.split(rng)
                    correct, total, rng = evaluate(state, utils.convert_to_jax(batch), subrng, head_id=head_id)
                    big_correct += correct
                    big_total += total

                eval_test_acc = big_correct / big_total
                del big_correct, big_total


                print(f"\nEpoch {epoch + 1}")
                print(f"Train Loss: {train_loss_train:.4f}")
                print(f"Train NLL: {train_nll_train:.4f}")
                print(f"Train KL: {train_kl_train:.4f}")
                print(f"Test Accuracy: {eval_test_acc:.4f}")
        
        print("\nUpdating prior parameters")
        state = train_state_replace(state)

        # fine-tune on coreset
        rng, subrng = jax.random.split(rng)
        if use_coreset:
            new_train_state = train_state_copy(state)
            for epoch in range(num_coreset_epochs):
                for batch in coreset_loaders[task_id]:
                    # if use_multiple_heads:
                    #     new_train_state = new_train_state.replace(curr_head=?)
                    batch = utils.convert_to_jax(batch)
                    rng, subrng = jax.random.split(rng)
                    new_train_state, _ = train_step(new_train_state, batch, subrng, kl_weight=1/(200 * (task_id+1)))
            
            for eval_task_id, (_, test_loader) in enumerate(data_loaders[:task_id+1]):
                if use_multiple_heads:
                    new_train_state = new_train_state.replace(curr_head=eval_task_id)
                print(f"\nTesting on Task with upgraded model trained on coreset {eval_task_id + 1}/{task_id+1}")
                rng, acc = eval_task(test_loader, new_train_state, rng, verbose=True)
        
        avg_acc = 0
        for eval_task_id, (_, test_loader) in enumerate(data_loaders[:task_id+1]):
            if use_multiple_heads:
                head_id = eval_task_id
            else:
                head_id = 0
            print(f"\nTesting on Task {eval_task_id + 1}/{task_id+1}")
            rng, acc = eval_task(test_loader, state, rng, head_id=head_id, verbose=True)
            avg_acc += acc
        avg_acc /= task_id+1
        avg_accuracies.append(avg_acc)
             
    return state, avg_accuracies



def get_dataloaders(
    task: str,
    batch_size: int = 128,
    num_tasks: int = 5
) -> Tuple[List[Tuple[DataLoader, DataLoader]], int]:
    if task == 'mnist_perm':
        return get_dataloaders_mnist_perm(batch_size, num_tasks), 1
    elif task == 'mnist_split':
        return get_dataloaders_mnist_split(batch_size, num_tasks), num_tasks
    
def get_coreset_loader(
    batch_size: int,
    data_loaders,
    coreset_size: int = 200
) -> Tuple[DataLoader, DataLoader]:
    return utils.get_coreset_dataloader(data_loaders, batch_size, coreset_size)


if __name__ == "__main__":
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(123)
    np.random.seed(123)

    model = VariationalMLP(num_classes=10)
    # Create and train model
    state, avg_accuracies = train_continual(
        model=model,
        task='mnist_perm',
        num_tasks=5,
        num_epochs=10,
        batch_size=256,
        learning_rate=0.001,
        use_coreset=True,
        num_coreset_epochs=100,
        rng=rng
    )

    print(f"Average accuracies: {avg_accuracies}")