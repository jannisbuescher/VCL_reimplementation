import torch
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
from typing import Tuple, List

def convert_to_jax(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert a PyTorch batch to JAX arrays.
    
    Args:
        batch: Tuple of (images, labels) from PyTorch DataLoader
    
    Returns:
        Tuple of (images, labels) as JAX arrays
    """
    images, labels = batch
    images = jnp.array(images.numpy())
    labels = jnp.array(labels.numpy())
    return images, labels

class CoresetData(torch.utils.data.TensorDataset):
    def __init__(self, dataloaders: List[Tuple[DataLoader, DataLoader]], coreset_size: int = 200):
        super().__init__()
        self.data = {}

        coreset_data = []
        coreset_labels = []
        coresets_loaders = []
        for task_id, (dataloader, _) in enumerate(dataloaders):
            coreset_size_remaining = coreset_size
            for batch, labels in dataloader:
                if len(batch) < coreset_size_remaining:
                    coreset_data.append(batch)
                    coreset_labels.append(labels)
                    coreset_size_remaining -= len(batch)
                else:
                    coreset_data.append(batch[:coreset_size_remaining])
                    coreset_labels.append(labels[:coreset_size_remaining])
                    break
            self.data[task_id] = torch.utils.data.TensorDataset(torch.cat(coreset_data), torch.cat(coreset_labels))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx: int, task_id: int = 0):
        return self.data[task_id][idx]
    
class CoresetDataLoader(torch.utils.data.TensorDataset):
    def __init__(self, coresets: List[CoresetData], batch_size: int):
        super().__init__()
        self.coresets = coresets
        self.batch_size = batch_size

    def __len__(self):
        return len(self.coresets[0]) * len(self.coresets)
    
    def __getitem__(self, idx: int):
        task_id = idx // self.batch_size
        idx = idx % self.batch_size
        return self.coresets[task_id][idx], task_id

def get_coreset_dataloader(dataloaders: List[Tuple[DataLoader, DataLoader]], batch_size: int, coreset_size: int = 200):
    coreset_data = []
    coreset_labels = []
    coresets_loaders = []
    for task_id, (dataloader, _) in enumerate(dataloaders):
        coreset_size_remaining = coreset_size
        for batch, labels in dataloader:
            if len(batch) < coreset_size_remaining:
                coreset_data.append(batch)
                coreset_labels.append(labels)
                coreset_size_remaining -= len(batch)
            else:
                coreset_data.append(batch[:coreset_size_remaining])
                coreset_labels.append(labels[:coreset_size_remaining])
                break
    coreset_data = torch.cat(coreset_data)
    coreset_labels = torch.cat(coreset_labels)
    for idx in range(len(dataloaders)):

        coreset = torch.utils.data.TensorDataset(coreset_data[:idx+1], coreset_labels[:idx+1])
        coreset_loader = DataLoader(
            coreset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        coresets_loaders.append(coreset_loader)
    return coresets_loaders