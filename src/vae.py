import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional
from model import VariationalDense

class VAE(nn.Module):
    """Variational Autoencoder with variational dense layers."""
    hidden_dim: int = 500
    latent_dim: int = 50
    
    def setup(self):
        # Encoder layers
        self.encoder_hidden = VariationalDense(28*28, self.hidden_dim)
        self.encoder_mu = VariationalDense(self.hidden_dim, self.latent_dim)
        self.encoder_var = VariationalDense(self.hidden_dim, self.latent_dim)
        
        # Decoder layers
        self.decoder_hidden = VariationalDense(self.latent_dim, self.hidden_dim)
        self.decoder_output = VariationalDense(self.hidden_dim, 28*28)
    
    def encode(self, x: jnp.ndarray, rng: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode input to latent space."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        rng, rng_hidden, rng_mu, rng_var = jax.random.split(rng, 4)
        
        # Hidden layer
        h = self.encoder_hidden(x, rng_hidden)
        h = nn.relu(h)
        
        # Mean and variance of latent space
        mu = self.encoder_mu(h, rng_mu)
        log_var = self.encoder_var(h, rng_var)
        
        return mu, log_var
    
    def reparameterize(self, mu: jnp.ndarray, log_var: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        """Reparameterization trick."""
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, shape=mu.shape)
        return mu + eps * std
    
    def decode(self, z: jnp.ndarray, rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Decode from latent space to input space."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        rng, rng_hidden, rng_output = jax.random.split(rng, 3)
        
        # Hidden layer
        h = self.decoder_hidden(z, rng_hidden)
        h = nn.relu(h)
        
        # Output layer
        x_recon = self.decoder_output(h, rng_output)
        return jax.nn.sigmoid(x_recon)
    
    def __call__(self, x: jnp.ndarray, rng: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass through the VAE."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        rng, rng_reparam = jax.random.split(rng)
        
        # Encode
        mu, log_var = self.encode(x, rng)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var, rng_reparam)
        
        # Decode
        x_recon = self.decode(z, rng)
        
        return x_recon, mu, log_var