import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
from typing import List
import numpy as np

def get_train_dataset_t1(root_dir : str, loader, transform: List):
    train_dataset = DatasetFolder(root=root_dir,extensions=".npy", loader=loader,transform=transform)
    return train_dataset


def get_val_dataset_t1(root_dir : str, loader, transform: List):
    val_dataset = DatasetFolder(root=root_dir,extensions=".npy", loader=loader,transform=transform)
    return val_dataset

def sample_loader(f :str):
    return np.load(f)

if __name__=="__main__":
    transform =transforms.Compose([   
        transforms.ToTensor()
    ])
    train_dataset = get_train_dataset_t1('/media/saitomar/Work/Projects/DeepLense_Test/task_1_dataset/dataset/train', sample_loader, transform=transform)
    print(len(train_dataset))
    # x = np.load("/media/saitomar/Work/Projects/DeepLense_Test/task_1_dataset/dataset/train/no/1.npy")
    # print(type(x[0][0][0]))
