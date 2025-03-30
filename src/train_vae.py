import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict

from vae import VAE
from mnist_perm import create_data_loaders as get_dataloaders_mnist_perm
from loss import variational_loss_vae, vae_loss
import utils

def create_data_loaders(batch_size: int = 128, num_tasks: int = 5):
    """Create MNIST data loader."""
    return get_dataloaders_mnist_perm(batch_size=batch_size, num_tasks=num_tasks)



class TrainState(train_state.TrainState):
    """Training state for the variational model."""
    prior_params: Dict
    rng: jax.random.PRNGKey


def create_train_state(
    model: VAE,
    learning_rate: float = 0.001,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> TrainState:
    """Create initial training state."""
    params = model.init(rng, jnp.ones((1, 28*28)))['params']
    prior_params = params

    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        prior_params=prior_params,
        tx=tx,
        rng=rng
    )

@jax.jit
def train_step(
    state: train_state.TrainState,
    images: Tuple[jnp.ndarray],
    rng: jax.random.PRNGKey,
    kl_weight: float = 1e-6
) -> Tuple[train_state.TrainState, Dict]:
    """Single training step."""
    
    def loss_fn(params):
        x_recon, mu, log_var = state.apply_fn({'params': params}, images, rng)
        loss, metrics = variational_loss_vae(
                params=params,
                prior_params=state.prior_params,
                x=images,
                x_recon=x_recon,
                mu=mu,
                log_var=log_var,
                kl_weight=kl_weight,
            )
        return loss, metrics
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    return state, metrics

def train_vae(
    num_epochs: int = 10,
    num_tasks: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
):
    """Train the VAE on MNIST."""
    # Create data loader
    data_loaders = create_data_loaders(batch_size)
    
    # Initialize model and training state
    model = VAE(hidden_dim=500, latent_dim=50)
    state = create_train_state(model, learning_rate, rng)

    
    # Training loop

    for task_id, (train_loader, test_loader) in enumerate(data_loaders):
        print(f'Task {task_id}')
            
        kl_weight = 1 / (batch_size * len(data_loaders[0]))

        for epoch in range(num_epochs):
        
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                # Convert batch to JAX format
                images, _ = utils.convert_to_jax(batch)
                
                # Training step
                rng, step_rng = jax.random.split(rng)
                state, metrics = train_step(state, images, step_rng, kl_weight=kl_weight)
                
                epoch_loss += metrics['total_loss']
            
            # Print epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

        # Test loop
        test_loss_vae = 0
        test_loss_mse = 0
        for batch in tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, _ = utils.convert_to_jax(batch)
            
            rng, step_rng = jax.random.split(rng)
            x_recon, mu, log_var = state.apply_fn({'params': state.params}, images, step_rng)
            
            loss = vae_loss(images, x_recon, mu, log_var)
            mse_loss = jnp.mean((images - x_recon) ** 2)
            test_loss_vae += loss
            test_loss_mse += mse_loss
            
        avg_loss_vae = test_loss_vae / len(test_loader)
        avg_loss_mse = test_loss_mse / len(test_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss_vae:.4f}, MSE Loss: {avg_loss_mse:.4f}')
                
    
    return state

if __name__ == "__main__":
    # Train the VAE
    rng = jax.random.PRNGKey(0)
    state = train_vae(
        num_epochs=10,
        num_tasks=3,
        batch_size=128,
        learning_rate=0.001,
        rng=rng
    ) 