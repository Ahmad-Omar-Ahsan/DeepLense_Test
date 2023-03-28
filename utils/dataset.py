from torchvision import transforms
import os
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, Subset
from torchvision.datasets import DatasetFolder
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from astropy.io import fits
import re
import torch


def sample_loader(f: str):
    return (
        np.load(
            f,
        )
        .astype("float32")
        .transpose(1, 2, 0)
    )


def get_dataset_t1(root_dir: str, loader=sample_loader):
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(100)])
    dataset = DatasetFolder(
        root=root_dir, extensions=".npy", loader=loader, transform=transform
    )
    return dataset


def get_train_val_dataloader_t1(root_dir: str, batch_size: int = 16):
    train_val_dir = os.path.join(root_dir, "train")

    train_val_dataset = get_dataset_t1(train_val_dir, loader=sample_loader)
    targets = train_val_dataset.targets
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=0.05, shuffle=True, stratify=targets
    )

    # Define sampler for train and validation sets
    train_ds = Subset(train_val_dataset, indices=train_idx)
    val_ds = Subset(train_val_dataset, indices=val_idx)

    # Define data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader


def get_test_dataloader_t1(root_dir: str, batch_size: int = 16):
    test_dir = os.path.join(root_dir, "val")
    test_dataset = get_dataset_t1(test_dir, loader=sample_loader)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return test_loader


class FitDataset(Dataset):
    def __init__(self, folder_path: str, label_file: str):

        self.folder_path = folder_path
        self.label_file = label_file
        self.filenames = os.listdir(self.folder_path)
        self.csv_file = pd.read_csv(self.label_file)
        self.labels = self.csv_file["is_lens"].values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        filename = os.path.join(self.folder_path, self.filenames[index])
        ID = re.findall("\d+", self.filenames[index])[0]

        df = self.csv_file.loc[self.csv_file["ID"] == int(ID)]
        label = df["is_lens"].values

        with fits.open(filename) as hdul:
            data = hdul[0].data

        data = data.astype(np.float32)
        tensor_data = torch.from_numpy(data).float().unsqueeze(dim=0)

        return tensor_data, label[0]


def get_dataset_t2(dataset_dir: str, batch_size: int = 16):
    training_dir = os.path.join(dataset_dir, "files")
    label_file = os.path.join(dataset_dir, "classifications.csv")
    ds = FitDataset(training_dir, label_file=label_file)

    # Split ratio 90-10 for train-test
    train_idx, test_idx = train_test_split(
        np.arange(len(ds)), test_size=0.1, shuffle=True, stratify=ds.labels
    )

    train_labels = ds.labels

    # Split ratio for new train set into train and val 95-5
    train_idx, val_idx = train_test_split(
        np.arange(len(train_idx)),
        test_size=0.05,
        shuffle=True,
        stratify=train_labels[train_idx],
    )

    train_ds = Subset(ds, indices=train_idx)
    val_ds = Subset(ds, indices=val_idx)
    test_ds = Subset(ds, indices=test_idx)

    # print(f"train_ds : {len(train_ds)}, val_ds: {len(val_ds)}, test_ds: {len(test_ds)}")
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    root_dir = "/media/saitomar/Work/Projects/DeepLense_Test/task_2_dataset"
    # dataset = get_dataset_t1(root_dir=root_dir)
    # print(dataset.samples)

    # transform=None
    train_dataloader, val_dataloader, test_dataloader = get_dataset_t2(
        dataset_dir=root_dir
    )
