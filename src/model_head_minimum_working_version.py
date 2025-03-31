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
        self.weights_mu = self.param('weights_mu', self.mu_init, (self.features_in, self.features_out))
        self.weights_var = self.param('weights_var', self.var_init, (self.features_in, self.features_out))
        
        self.bias_mu = self.param('bias_mu', self.mu_init, (self.features_out,))
        self.bias_var = self.param('bias_var', self.var_init, (self.features_out,))

    def __call__(self, x: jnp.ndarray, rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        rng, rng_weights, rng_bias = jax.random.split(rng, 3)
        weights_eps = jax.random.normal(rng_weights, shape=self.weights_mu.shape)
        bias_eps = jax.random.normal(rng_bias, shape=self.bias_mu.shape)

        # jax.debug.callback(lambda x, y: print("\n\nweights_var", x, y), self.weights_var, jnp.exp(0.5 * self.weights_var))

        
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
        if rng is None:
            rng = jax.random.PRNGKey(0)
            
        rng, rng_first, rng_second, rng_third = jax.random.split(rng, 4)
        # jax.debug.callback(lambda x: print("call"), None)

        # jax.debug.callback(lambda x: print(x), x)

        x = VariationalDense(28*28,self.hidden_dims[0])(x, rng_first)
        x = nn.relu(x)

        # jax.debug.callback(lambda x: print(x), x)
        
        x = VariationalDense(self.hidden_dims[0], self.hidden_dims[1])(x, rng_second)
        x = nn.relu(x)


        # jax.debug.callback(lambda x: print(x), x)

        def head_fn(i):
            return lambda mdl, x, key: mdl.heads[i](x, key)
        branches = [head_fn(i) for i in range(len(self.heads))]
        if self.is_mutable_collection('params'):
            for branch in branches:
                _ = branch(self, x, rng_third)
        if head_id is None:
            head_id = 0
        x = nn.switch(head_id, branches, self, x, rng_third)

        # jax.debug.callback(lambda x: print(x), x)

        x = nn.softmax(x, axis=-1)


        # jax.debug.callback(lambda x: print(x), x)


        return x