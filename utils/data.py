"""
Utility functions to load and transform datasets
"""
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
from torch.utils.data import TensorDataset
import os
import numpy as np

def load_data(data_name, n_df, root='./', device='cpu', split='train'):
    os.makedirs(f'{root}/data', exist_ok=True)
    DATA_ROOT = f'{root}/data/{data_name}'

    if data_name == 'mnist':
        # load MNIST dataset
        MNIST_MEAN = (0.1307,)
        MNIST_STD_DEV = (0.3081,)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD_DEV)
        ])

        dataset = MNIST(
            root=DATA_ROOT, train=split=='train', download=True, transform=transform)

        out_dim = 10
    elif data_name == 'cifar10':
        # load CIFAR-10 dataset
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
        ])

        dataset = CIFAR10(
            root=DATA_ROOT, train=split=='train', download=True, transform=transform)
        
        out_dim = 10
    elif data_name == 'cifar100':
        # load CIFAR-100 dataset
        CIFAR100_MEAN = (0.5074, 0.4867, 0.4411)
        CIFAR100_STD_DEV = (0.2011, 0.1987, 0.2025)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD_DEV),
        ])

        dataset = CIFAR100(
            root=DATA_ROOT, train=split=='train', download=True, transform=transform)
        
        out_dim = 100
    elif os.path.exists(f'{DATA_ROOT}'):
        # load pre-processed local data
        X, y = np.load(f'{DATA_ROOT}/X_{split}.npy'), np.load(f'{DATA_ROOT}/y_{split}.npy')
        X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)

        dataset = TensorDataset(X, y)
        out_dim = len(y.unique())
    else:
        raise Exception(f'Dataset {data_name} not configured and {DATA_ROOT} not found')
    
    # load neighboring dataset D-
    n_df = len(dataset) if n_df is None or n_df < 0 else n_df # load full dataset if n_df is None
    shuffle = n_df != len(dataset) # only shuffle dataset if full dataset is not loaded
    tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=n_df, shuffle=shuffle)

    X, y = next(iter(tmp_loader))
    X, y = X.to(device), y.to(device)

    return X, y, out_dim