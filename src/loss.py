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

    # jax.debug.callback(
    #     lambda x, y,z: print(x,y,z ), 
    #     jnp.sum(jnp.log(sigma_p) - jnp.log(sigma_q)), 
    #     jnp.sum(sigma_q), 
    #     jnp.sum(sigma_p))
    # jax.debug.callback(lambda x: print(x), jnp.sum(sigma_q / sigma_p))
    # jax.debug.callback(lambda x: print(x), jnp.sum(jnp.square(mu_p - mu_q) / sigma_p))
    # jax.debug.callback(lambda x:print(x), sigma_q.shape[0])

    term1 = 0.5 * jnp.sum(jnp.log(sigma_p / sigma_q))
    term2 = 0.5 * jnp.sum(sigma_q / sigma_p)
    term3 = 0.5 * jnp.sum(jnp.square(mu_p - mu_q) / sigma_p)
    term4 = -0.5 * sigma_q.shape[0]

    # jax.debug.callback(lambda x,y,z,w: print(x,y,z,w), term1, term2, term3, term4)

    kl = term1 + term2 + term3 + term4

    # kl = 0.5 * jnp.sum(jnp.log(sigma_q) - jnp.log(sigma_p)
    #                    + ((sigma_q + jnp.square(mu_p - mu_q)) / sigma_p)
    #                    - 1)
    return kl

    # var_q = jnp.exp(log_sigma_q)
    # var_p = jnp.exp(log_sigma_p)
    # mu_diff_squared = jnp.square(mu_q - mu_p)
    # quot_term = var_q / var_p
    # log_diff_term = (log_sigma_p - log_sigma_q)
    # kl = quot_term 
    # kl = kl + (mu_diff_squared / var_p)
    # kl = kl - 1
    # kl = kl + log_diff_term
    # kl = 0.5 * jnp.sum(kl)
    
    # return kl



    # sigma_q = jnp.exp(log_sigma_q)
    # sigma_p = jnp.exp(log_sigma_p)
    # return 0.5 * jnp.sum(
    #     (sigma_q / sigma_p) +
    #     (jnp.square(mu_p - mu_q) / sigma_p) -
    #     1 +
    #     (log_sigma_p - log_sigma_q)
    # )

def variational_loss(
    params: Dict,
    prior_params: Dict,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    kl_weight: float = 1e-6,
    compute_all_kl: bool = True
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute the variational loss: negative log likelihood + KL divergence.
    
    Args:
        params: Current model parameters (posterior)
        prior_params: Prior parameters
        logits: Model output logits
        labels: Class indices (not one-hot encoded)
        num_samples: Number of Monte Carlo samples for the expectation
    
    Returns:
        total_loss: Combined loss value
        metrics: Dictionary containing individual loss components
    """
    # Compute cross entropy loss (negative log likelihood)
    nll = jnp.sum(jnp.multiply(jax.nn.one_hot(labels, 10), jnp.log(logits + 1e-9)), axis=1)
    nll = -jnp.sum(nll)

    # Compute KL divergence for each layer
    kl_div = 0.0
    for layer_name in params.keys():
        if compute_all_kl or 'Dense' in layer_name:
            # Compute KL for weights
            kl_div += gaussian_kl_divergence(
                mu_q=jnp.ravel(params[layer_name]['weights_mu']),
                log_sigma_q=jnp.ravel(params[layer_name]['weights_var']),
                mu_p=jnp.ravel(prior_params[layer_name]['weights_mu']),
                log_sigma_p=jnp.ravel(prior_params[layer_name]['weights_var'])
            )
            
            # Compute KL for biases
            kl_div += gaussian_kl_divergence(
                mu_q=jnp.ravel(params[layer_name]['bias_mu']),
                log_sigma_q=jnp.ravel(params[layer_name]['bias_var']),
                mu_p=jnp.ravel(prior_params[layer_name]['bias_mu']),
                log_sigma_p=jnp.ravel(prior_params[layer_name]['bias_var'])
            )
    
    total_loss = nll + kl_weight*kl_div
    
    metrics = {
        'nll': nll,
        'kl_div': kl_div,
        'total_loss': total_loss
    }
    
    return total_loss, metrics 


def vae_loss(x: jnp.ndarray, x_recon: jnp.ndarray, mu: jnp.ndarray, log_var: jnp.ndarray) -> jnp.ndarray:
    """Compute VAE loss (reconstruction + KL divergence)."""
    # Reconstruction loss (binary cross-entropy)
    bce = -jnp.sum(x * jnp.log(x_recon + 1e-8) + (1 - x) * jnp.log(1 - x_recon + 1e-8))
    
    # KL divergence
    kl_div = -0.5 * jnp.sum(1 + log_var - jnp.square(mu) - jnp.exp(log_var))
    
    return bce + kl_div 

def variational_loss_vae(
    params: Dict,
    prior_params: Dict,
    x: jnp.ndarray,
    x_recon: jnp.ndarray,
    mu: jnp.ndarray,
    log_var: jnp.ndarray,
    kl_weight: float = 1e-6
) -> Tuple[jnp.ndarray, Dict]:
    kl_div = 0.0
    for layer_name in params.keys():
        kl_div += gaussian_kl_divergence(
            mu_q=jnp.ravel(params[layer_name]['weights_mu']),
            log_sigma_q=jnp.ravel(params[layer_name]['weights_var']),
            mu_p=jnp.ravel(prior_params[layer_name]['weights_mu']),
            log_sigma_p=jnp.ravel(prior_params[layer_name]['weights_var'])
        )

    vae_loss_val = vae_loss(x, x_recon, mu, log_var)
    total_loss = vae_loss_val + kl_weight*kl_div
    return total_loss, {'vae_loss': vae_loss_val, 'kl_div': kl_div, 'total_loss': total_loss}
    
        