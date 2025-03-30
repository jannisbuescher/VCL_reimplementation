import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple
import jax.numpy as jnp

class BinaryMNIST(Dataset):
    """Custom dataset for binary classification tasks from MNIST."""
    def __init__(self, data: torch.Tensor, digit1: int, digit2: int):
        train_data = data.data
        train_labels = data.targets

        # Create boolean masks for labels 0 and 1
        train_mask = (train_labels == digit1) | (train_labels == digit2)
        self.data = train_data[train_mask]
        self.targets = train_labels[train_mask]

        # # Filter for only the two digits we want
        # data = [(x, y) for x, y in data if y == digit1 or y == digit2]
        # self.data = torch.stack([x[0] for x in data])
        # self.targets = torch.tensor([x[1] for x in data])
        
        # Convert targets to binary (0 for digit1, 1 for digit2)
        self.targets = (self.targets == digit2).long()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].view(-1), self.targets[idx]

def get_dataloaders(
    batch_size: int = 128,
    num_tasks: int = 5,
) -> List[DataLoader]:
    """
    Create 5 binary classification tasks from MNIST.
    Each task is a binary classification
    
    Args:
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        List of training dataloaders for each task
    """
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load full MNIST dataset
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create 5 binary classification tasks
    tasks = []
    for i in range(0, 10, 2):
        # Create training dataset for this task
        train_dataset = BinaryMNIST(
            train_data,
            i, i+1
        )

        test_dataset = BinaryMNIST(
            test_data,
            i, i+1
        )
        
        # Create dataloader
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
            num_workers=0,
            pin_memory=True
        )

        tasks.append((train_loader, test_loader))
    
    return tasks[:num_tasks]

if __name__ == "__main__":
    tasks = get_dataloaders(batch_size=128, num_workers=0)
    for i, (train_loader, test_loader) in enumerate(tasks):
        print(f"Task {i+1}:")
        print(f"Training batches: {len(train_loader)}")
        print(f"Testing batches: {len(test_loader)}")
