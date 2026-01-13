# Prepare MNIST data split across sites
import os
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

def prepare_data(num_sites):
    full_train = datasets.MNIST('/tmp/flare/dataset/', train=True, download=True, transform=transforms.ToTensor())
    full_test = datasets.MNIST('/tmp/flare/dataset/', train=False, download=True, transform=transforms.ToTensor())
    
    train_sizes = [len(full_train) // num_sites] * (num_sites - 1) + [len(full_train) - (len(full_train) // num_sites) * (num_sites - 1)]
    test_sizes = [len(full_test) // num_sites] * (num_sites - 1) + [len(full_test) - (len(full_test) // num_sites) * (num_sites - 1)]
    
    train_splits = random_split(full_train, train_sizes)
    test_splits = random_split(full_test, test_sizes)
    
    for i in range(num_sites):
        site_id = i + 1  # 1-indexed site folders
        site_dir = f'/tmp/mnist_experiment/site-{site_id}'
        os.makedirs(site_dir, exist_ok=True)
        torch.save({'train_indices': train_splits[i].indices, 'test_indices': test_splits[i].indices}, f'{site_dir}/indices.pt')
        print(f"Site {site_id}: {len(train_splits[i])} train, {len(test_splits[i])} test samples")

if __name__ == '__main__':
    prepare_data(num_sites=3)
