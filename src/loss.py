import jax
import jax.numpy as jnp
from typing import Dict, Tuple

def gaussian_kl_divergence(
    mu_q: jnp.ndarray,
    sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    sigma_p: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute KL divergence between two Gaussian distributions.
    KL(q||p) = 1/2 * sum(sigma_q^2/sigma_p^2 + (mu_p - mu_q)^2/sigma_p^2 - 1 + log(sigma_p^2/sigma_q^2))
    """
    return 0.5 * jnp.sum(
        sigma_q / sigma_p +
        (mu_p - mu_q) ** 2 / sigma_p -
        1 +
        jnp.log(sigma_p / sigma_q)
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
    labels: jnp.ndarray
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
    nll = cross_entropy_loss(logits, labels)
    
    # Compute KL divergence for each layer
    kl_div = 0.0
    for layer_name in params.keys():
        if 'Dense' in layer_name:
            # Compute KL for weights
            kl_div += gaussian_kl_divergence(
                mu_q=params[layer_name]['weights_mu'],
                sigma_q=jnp.exp(params[layer_name]['weights_var']),
                mu_p=prior_params[layer_name]['weights_mu'],
                sigma_p=jnp.exp(prior_params[layer_name]['weights_var'])
            )
            
            # Compute KL for biases
            kl_div += gaussian_kl_divergence(
                mu_q=params[layer_name]['bias_mu'],
                sigma_q=jnp.exp(params[layer_name]['bias_var']),
                mu_p=prior_params[layer_name]['bias_mu'],
                sigma_p=jnp.exp(prior_params[layer_name]['bias_var'])
            )
    
    # Total loss is NLL + KL divergence
    total_loss = nll + kl_div
    
    metrics = {
        'nll': nll,
        'kl_div': kl_div,
        'total_loss': total_loss
    }
    
    return total_loss, metrics 