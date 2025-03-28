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
from .mnist_perm import create_data_loaders, convert_to_jax

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

def evaluate(
    state: TrainState,
    test_loader: List[Tuple[jnp.ndarray, jnp.ndarray]],
    num_samples: int = 10
) -> Dict:
    """Evaluate the model on test data."""
    metrics_list = []
    correct = 0
    total = 0
    
    for batch in test_loader:
        images, labels = convert_to_jax(batch)
        
        # Monte Carlo sampling for prediction
        logits_samples = []
        for _ in range(num_samples):
            logits = state.apply_fn({'params': state.params}, images, state.rng)
            logits_samples.append(logits)
        
        # Average predictions
        logits = jnp.mean(jnp.stack(logits_samples), axis=0)
        predictions = jnp.argmax(logits, axis=-1)
        
        # Compute accuracy
        correct += jnp.sum(predictions == labels)
        total += len(labels)
        
        # Compute loss
        _, metrics = variational_loss(
            params=state.params,
            prior_params=state.prior_params,
            logits=logits,
            labels=labels,
            num_samples=num_samples
        )
        metrics_list.append(metrics)
    
    # Average metrics
    avg_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
        for k in metrics_list[0].keys()
    }
    avg_metrics['accuracy'] = correct / total
    
    return avg_metrics

def train_continual(
    model: VariationalMLP,
    num_tasks: int = 5,
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    num_samples: int = 1,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
):
    """Train the model on multiple tasks sequentially."""
    # Create data loaders for all tasks
    data_loaders = create_data_loaders(
        batch_size=batch_size,
        num_tasks=num_tasks
    )
    
    # Initialize training state
    state = create_train_state(model, learning_rate, rng)
    
    # Train on each task
    for task_id, (train_loader, test_loader) in enumerate(data_loaders):
        print(f"\nTraining on Task {task_id + 1}/{num_tasks}")
        
        for epoch in range(num_epochs):
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
            
            # Evaluate on test set
            test_metrics = evaluate(state, test_loader, num_samples)
            
            # Print metrics every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch + 1}")
                print(f"Train Loss: {avg_metrics['total_loss']:.4f}")
                print(f"Test Loss: {test_metrics['total_loss']:.4f}")
                print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        # Update prior parameters with current posterior
        state = state.replace(prior_params=state.params)
    
    return state

if __name__ == "__main__":
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(42)
    np.random.seed(42)
    
    # Create and train model
    model = VariationalMLP()
    state = train_continual(
        model=model,
        num_tasks=5,
        num_epochs=10,
        batch_size=128,
        learning_rate=0.001,
        num_samples=1,
        rng=rng
    ) 