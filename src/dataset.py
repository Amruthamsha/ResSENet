"""FER2013 dataset loading and augmentation pipeline."""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

from .config import DEFAULT_CONFIG

# Paper stats: mu=0.507, sigma=0.255 (training set)
MU, SIGMA = 0.507, 0.255


def build_transforms(img_size: int = DEFAULT_CONFIG.img_size):
    train_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, scale=(0.9, 1.1)),
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            T.ColorJitter(brightness=0.25, contrast=0.25),
            T.ToTensor(),
            T.Normalize(mean=[MU, MU, MU], std=[SIGMA, SIGMA, SIGMA]),
            T.RandomErasing(p=0.5, scale=(0.01, 0.04), ratio=(1.0, 1.0)),
        ]
    )
    eval_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[MU, MU, MU], std=[SIGMA, SIGMA, SIGMA]),
        ]
    )
    return train_transform, eval_transform


class FER2013Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.labels = df["emotion"].values.astype(np.int64)
        # pixels are space-separated strings
        self.images = np.array(
            [np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in df["pixels"].values]
        )
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_rgb = np.stack([img, img, img], axis=2)  # grayscale -> RGB (channel replicate)
        if self.transform:
            img_rgb = self.transform(img_rgb)
        return img_rgb, self.labels[idx]


def load_fer2013(csv_path: str):
    """Load fer2013.csv and split into the standard Training/PublicTest/PrivateTest splits."""
    df_raw = pd.read_csv(csv_path)
    train_df = df_raw[df_raw["Usage"] == "Training"].reset_index(drop=True)
    val_df = df_raw[df_raw["Usage"] == "PublicTest"].reset_index(drop=True)
    test_df = df_raw[df_raw["Usage"] == "PrivateTest"].reset_index(drop=True)
    return train_df, val_df, test_df


def build_loaders(train_df, val_df, test_df, config: DEFAULT_CONFIG.__class__ = DEFAULT_CONFIG, device=None):
    """Build train/val/test DataLoaders plus class-weight tensor for weighted CE loss."""
    train_transform, eval_transform = build_transforms(config.img_size)

    train_ds = FER2013Dataset(train_df, train_transform)
    val_ds = FER2013Dataset(val_df, eval_transform)
    test_ds = FER2013Dataset(test_df, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(config.num_classes), y=train_df["emotion"].values
    )
    class_weights_tensor = torch.FloatTensor(class_weights)
    if device is not None:
        class_weights_tensor = class_weights_tensor.to(device)

    return train_loader, val_loader, test_loader, class_weights_tensor
