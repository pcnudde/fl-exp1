# Federated learning with multiprocessing for parallel site training
import torch
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))

def train_site(args):
    site_id, global_state_dict = args
    
    # Load site data
    full_train = datasets.MNIST('/tmp/flare/dataset/', train=True, download=False, transform=transforms.ToTensor())
    full_test = datasets.MNIST('/tmp/flare/dataset/', train=False, download=False, transform=transforms.ToTensor())
    indices = torch.load(f'/tmp/mnist_experiment/site-{site_id}/indices.pt')
    train_loader = DataLoader(Subset(full_train, indices['train_indices']), batch_size=64, shuffle=True)
    test_loader = DataLoader(Subset(full_test, indices['test_indices']), batch_size=1000)
    
    # Train
    model = SimpleNet()
    model.load_state_dict(global_state_dict)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(data), target)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        correct = sum((model(data).argmax(1) == target).sum().item() for data, target in test_loader)
    accuracy = 100 * correct / len(indices['test_indices'])
    
    print(f"Site {site_id} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    return model.state_dict()

def federated_training():
    mp.set_start_method('spawn')
    num_sites, num_rounds = 3, 5
    
    global_model = SimpleNet()
    
    with mp.Pool(num_sites) as pool:
        for round_num in range(num_rounds):
            print(f"\nRound {round_num + 1}/{num_rounds}")
            
            site_state_dicts = pool.map(train_site, [(i, global_model.state_dict()) for i in range(1, num_sites + 1)])
            global_model.load_state_dict({k: torch.stack([sd[k] for sd in site_state_dicts]).mean(0) for k in site_state_dicts[0]})
            
            # Evaluate global model
            test_loader = DataLoader(datasets.MNIST('/tmp/flare/dataset/', train=False, transform=transforms.ToTensor()), batch_size=1000)
            with torch.no_grad():
                correct = sum((global_model(d).argmax(1) == t).sum().item() for d, t in test_loader)
            print(f"Global Accuracy: {100 * correct / 10000:.2f}%")

if __name__ == '__main__':
    federated_training()
