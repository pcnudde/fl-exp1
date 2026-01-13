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
    
    for site_id in range(num_sites):
        site_dir = f'/tmp/mnist_experiment/site-{site_id}'
        os.makedirs(site_dir, exist_ok=True)
        torch.save({'train_indices': train_splits[site_id].indices, 'test_indices': test_splits[site_id].indices}, f'{site_dir}/indices.pt')
        print(f"Site {site_id}: {len(train_splits[site_id])} train, {len(test_splits[site_id])} test samples")

if __name__ == '__main__':
    prepare_data(num_sites=3)
