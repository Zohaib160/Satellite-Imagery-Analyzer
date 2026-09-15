import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from typing import Tuple
from src.config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


class TransformWrapper(Dataset):
    """
    Wraps a Subset to apply a specific transform.
    Defined at module level so it can be pickled by multiprocessing workers.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class EuroSATDataset(Dataset):
    """
    EuroSAT dataset wrapper for satellite image classification.
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


def get_data_loaders(
    data_dir: str,
    batch_size: int,
    train_ratio: float,
    val_ratio: float
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the EuroSAT dataset into train, val, and test dataloaders.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Load dataset without transforms to get total length for splitting
    base_dataset = datasets.ImageFolder(root=data_dir)
    total_len = len(base_dataset)

    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    train_ds, val_ds, test_ds = random_split(
        base_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap each split with appropriate transforms
    train_dataset = TransformWrapper(train_ds, train_transform)
    val_dataset = TransformWrapper(val_ds, val_test_transform)
    test_dataset = TransformWrapper(test_ds, val_test_transform)

    # num_workers=0 avoids multiprocessing issues on macOS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
