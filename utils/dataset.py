import torch
import os
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision.datasets import DatasetFolder
from typing import List
import numpy as np


def sample_loader(f :str):
    return np.load(f)

def get_dataset_t1(root_dir : str, loader = sample_loader, transform: List = None):
    train_dataset = DatasetFolder(root=root_dir,extensions=".npy", loader=loader,transform=transform)
    return train_dataset



def get_train_val_dataloader_t1(root_dir : str, batch_size : int = 16):
    train_val_dir = os.path.join(root_dir, 'train')

    train_val_dataset = get_dataset_t1(train_val_dir,loader=sample_loader)
    num_data = len(train_val_dataset)
    indices = list(range(num_data))
    split = int(num_data * 0.95) # 95% for training, 5% for validation
    train_indices, val_indices = indices[:split], indices[split:]

    # Define sampler for train and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Define data loaders
    train_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def get_test_dataloader_t1(root_dir : str, batch_size : int = 16):
    test_dir = os.path.join(root_dir, 'val')
    test_dataset = get_dataset_t1(test_dir, loader=sample_loader)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return test_loader




if __name__=="__main__":
    root_dir = '/media/saitomar/Work/Projects/DeepLense_Test/task_1_dataset/dataset'
    
    train_dataloader, val_dataloader = get_train_val_dataloader_t1(root_dir)
    test_dataloader = get_test_dataloader_t1(root_dir)
    
    for i, (sample, label) in enumerate(train_dataloader):
        print(sample.shape, label)
        print(sample)
        break
