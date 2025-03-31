import jax
import jax.numpy as jnp
from typing import Dict, Tuple

import optax

from functools import partial

def gaussian_kl_divergence(
    mu_q: jnp.ndarray,
    log_sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    log_sigma_p: jnp.ndarray
) -> jnp.ndarray:
    
    sigma_q = jnp.exp(log_sigma_q)
    sigma_p = jnp.exp(log_sigma_p)

    term1 = 0.5 * jnp.sum(jnp.log(sigma_p / sigma_q))
    term2 = 0.5 * jnp.sum(sigma_q / sigma_p)
    term3 = 0.5 * jnp.sum(jnp.square(mu_p - mu_q) / sigma_p)
    term4 = -0.5 * sigma_q.shape[0]

    # jax.debug.callback(lambda x,y,z,w: print(x,y,z,w), term1, term2, term3, term4)

    kl = term1 + term2 + term3 + term4
    return kl

def variational_loss(
    params: Dict,
    prior_params: Dict,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    head_id: int = 0,
    kl_weight: float = 1e-6
) -> Tuple[jnp.ndarray, Dict]:
    num_classes = logits.shape[1]
    nll = jnp.sum(jnp.multiply(jax.nn.one_hot(labels, num_classes), jnp.log(logits + 1e-9)), axis=1)
    nll = -jnp.sum(nll)
    
    kl_div = 0.0
    for layer_name in params.keys():
        if 'heads' not in layer_name or layer_name == f'heads_{head_id}':
            # weights
            kl_div += gaussian_kl_divergence(
                mu_q=jnp.ravel(params[layer_name]['weights_mu']),
                log_sigma_q=jnp.ravel(params[layer_name]['weights_var']),
                mu_p=jnp.ravel(prior_params[layer_name]['weights_mu']),
                log_sigma_p=jnp.ravel(prior_params[layer_name]['weights_var'])
            )
            # bias
            kl_div += gaussian_kl_divergence(
                mu_q=jnp.ravel(params[layer_name]['bias_mu']),
                log_sigma_q=jnp.ravel(params[layer_name]['bias_var']),
                mu_p=jnp.ravel(prior_params[layer_name]['bias_mu']),
                log_sigma_p=jnp.ravel(prior_params[layer_name]['bias_var'])
            )
    
    total_loss = nll + kl_weight * kl_div
    
    metrics = {
        'nll': nll,
        'kl_div': kl_div,
        'total_loss': total_loss
    }
    
    return total_loss, metrics 