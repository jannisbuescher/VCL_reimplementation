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
    """
    Compute KL divergence between two Gaussian distributions.

    """

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

    sigma_q = jnp.exp(log_sigma_q)
    sigma_p = jnp.exp(log_sigma_p)
    return 0.5 * jnp.sum(
        (sigma_q / sigma_p) +
        (jnp.square(mu_p - mu_q) / sigma_p) -
        1 +
        (log_sigma_p - log_sigma_q)
    )

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute cross entropy loss between logits and class indices."""
    # Convert labels to one-hot encoding
    num_classes = logits.shape[-1]
    labels_one_hot = jax.nn.one_hot(labels, num_classes)
    return -jnp.mean(jnp.sum(labels_one_hot * jnp.log(logits + 1e-7), axis=-1))

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
    nll = jnp.sum(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, 10))) #cross_entropy_loss(logits, labels)
    
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