from .model import VariationalMLP, VariationalDense
from .loss import variational_loss, cross_entropy_loss, gaussian_kl_divergence
from .mnist_perm import PermutedMNIST, create_data_loaders
from .train import train_continual, TrainState, create_train_state, train_step, evaluate

__version__ = "0.1.0"

__all__ = [
    "VariationalMLP",
    "VariationalDense",
    "variational_loss",
    "cross_entropy_loss",
    "gaussian_kl_divergence",
    "PermutedMNIST",
    "create_data_loaders",
    "convert_to_jax",
    "train_continual",
    "TrainState",
    "create_train_state",
    "train_step",
    "evaluate",
] 