import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import jax.numpy as jnp
import multiprocessing
import os

# Set multiprocessing start method to 'spawn' to avoid deadlocks
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['PYTHONUNBUFFERED'] = '1'

class NormalMNIST(Dataset):
    """Dataset class for normal MNIST."""
    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        download: bool = True
    ):
        self.mnist = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=torchvision.transforms.ToTensor()
        )

    def __len__(self) -> int:
        return len(self.mnist)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.mnist[idx]
        return image.view(-1), label



class PermutedMNIST(Dataset):
    """Dataset class for permuted MNIST."""
    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        permutation: Optional[np.ndarray] = None,
        download: bool = True
    ):
        """
        Initialize the permuted MNIST dataset.
        
        Args:
            root: Root directory for storing the dataset
            train: Whether to load training or test data
            permutation: Optional permutation to apply to the pixels
            download: Whether to download the dataset if it doesn't exist
        """
        self.mnist = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=torchvision.transforms.ToTensor()
        )
        
        # Generate random permutation if none provided
        if permutation is None:
            self.permutation = np.random.permutation(28*28)
        else:
            self.permutation = permutation
            
    def __len__(self) -> int:
        return len(self.mnist)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.mnist[idx]
        # Flatten and permute the image
        image = image.view(-1)[self.permutation]
        return image, label
    
    def get_permutation(self) -> np.ndarray:
        """Return the current permutation."""
        return self.permutation

def create_data_loaders(
    batch_size: int = 128,
    num_tasks: int = 5,
    root: str = './data'
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create data loaders for multiple permuted MNIST tasks.
    
    Args:
        batch_size: Batch size for the data loaders
        num_tasks: Number of different permutations to create
        root: Root directory for storing the dataset
    
    Returns:
        List of tuples containing (train_loader, test_loader) for each task
    """
    data_loaders = []

    transform_mnist = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset_normal = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform_mnist
    )
    test_dataset_normal = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform_mnist
    )

    train_loader_normal = DataLoader(
        train_dataset_normal,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader_normal = DataLoader(
        test_dataset_normal,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    data_loaders.append((train_loader_normal, test_loader_normal))

    for task_id in range(num_tasks - 1):
        # Generate a new permutation for each task
        permutation = np.random.permutation(784)
        
        # Create train and test datasets
        train_dataset = PermutedMNIST(
            root=root,
            train=True,
            permutation=permutation
        )
        test_dataset = PermutedMNIST(
            root=root,
            train=False,
            permutation=permutation
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        data_loaders.append((train_loader, test_loader))
    
    return data_loaders
