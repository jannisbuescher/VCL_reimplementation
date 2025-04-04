import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional, List

class VariationalDense(nn.Module):
    """A variational dense layer that maintains mean and variance for weights and biases."""
    features_in: int
    features_out: int
    mu_init: callable = nn.initializers.normal(stddev=0.1)
    var_init: callable = nn.initializers.constant(-13.0) 

    def setup(self):
        # jax.debug.callback(lambda x: print(x), ten_6)
        # Initialize mean and variance for weights
        self.weights_mu = self.param('weights_mu', self.mu_init, (self.features_in, self.features_out))
        self.weights_var = self.param('weights_var', self.var_init, (self.features_in, self.features_out))

        # Initialize mean and variance for bias
        self.bias_mu = self.param('bias_mu', self.mu_init, (self.features_out,))
        self.bias_var = self.param('bias_var', self.var_init, (self.features_out,))
        
    def __call__(self, x: jnp.ndarray, rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Forward pass with reparameterization trick."""      
        # Sample weights and biases using reparameterization trick
        rng, rng_weights, rng_bias = jax.random.split(rng, 3)
        weights_eps = jax.random.normal(rng_weights, shape=self.weights_mu.shape)
        bias_eps = jax.random.normal(rng_bias, shape=self.bias_mu.shape)
        
        weights = self.weights_mu + jnp.exp(0.5 * self.weights_var) * weights_eps
        bias = self.bias_mu + jnp.exp(0.5* self.bias_var) * bias_eps
        
        return jnp.matmul(x, weights) + bias

class VariationalMLP(nn.Module):
    """A variational MLP with three layers and softmax output."""
    hidden_dims: Tuple[int, int] = (100, 100)
    num_classes: int = 10
    num_heads: int = 5  # Number of different heads to create

    def setup(self):
        self.heads = [VariationalDense(self.hidden_dims[1], self.num_classes) for _ in range(self.num_heads)]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, rng: Optional[jnp.ndarray] = None, head_id: Optional[int] = None) -> jnp.ndarray:
        """Forward pass through the variational MLP."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
            
        rng, rng_first, rng_second, rng_third = jax.random.split(rng, 4)

        # First layer
        x = VariationalDense(28*28,self.hidden_dims[0])(x, rng_first)
        x = nn.relu(x)
        
        # Second layer
        x = VariationalDense(self.hidden_dims[0], self.hidden_dims[1])(x, rng_second)
        x = nn.relu(x)

        # # Third layer
        # x = VariationalDense(self.hidden_dims[1], self.hidden_dims[2])(x, rng_third)
        # x = nn.relu(x)

        # Head layer
        def head_fn(i):
            return lambda mdl, x, key: mdl.heads[i](x, key)
        branches = [head_fn(i) for i in range(len(self.heads))]

        # run all branches on init
        if self.is_mutable_collection('params'):
            for branch in branches:
                _ = branch(self, x, rng_third)

        if head_id is None:
            head_id = 0

        x = nn.switch(head_id, branches, self, x, rng_third)

        return nn.softmax(x, axis=-1)