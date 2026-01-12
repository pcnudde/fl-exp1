# Federated learning simulation with data split and saved to disk per site
import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))

def train_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(data), target)
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate(model, test_loader, test_dataset):
    model.eval()
    with torch.no_grad():
        correct = sum((model(data).argmax(1) == target).sum().item() for data, target in test_loader)
    return 100 * correct / len(test_dataset)

def average_weights(models):
    state_dicts = [model.state_dict() for model in models]
    return {key: torch.stack([sd[key] for sd in state_dicts]).mean(dim=0) for key in state_dicts[0].keys()}

def prepare_site_data(num_sites=3):
    full_train = datasets.MNIST('/tmp/flare/dataset/', train=True, download=True, transform=transforms.ToTensor())
    full_test = datasets.MNIST('/tmp/flare/dataset/', train=False, download=True, transform=transforms.ToTensor())
    
    train_size = len(full_train) // num_sites
    train_splits = random_split(full_train, [train_size] * (num_sites - 1) + [len(full_train) - train_size * (num_sites - 1)])
    
    test_size = len(full_test) // num_sites
    test_splits = random_split(full_test, [test_size] * (num_sites - 1) + [len(full_test) - test_size * (num_sites - 1)])
    
    for site_id in range(num_sites):
        site_dir = f'/tmp/mnist_experiment/site-{site_id}'
        os.makedirs(site_dir, exist_ok=True)
        train_indices = train_splits[site_id].indices
        test_indices = test_splits[site_id].indices
        torch.save({'train_indices': train_indices,'test_indices': test_indices}, f'{site_dir}/indices.pt')
        print(f"Site {site_id}: {len(train_indices)} train samples, {len(test_indices)} test samples saved to {site_dir}")
    

def load_site_data(site_id):
    site_dir = f'/tmp/mnist_experiment/site-{site_id}'
    full_train = datasets.MNIST('/tmp/flare/dataset/', train=True, download=False, transform=transforms.ToTensor())
    full_test = datasets.MNIST('/tmp/flare/dataset/', train=False, download=False, transform=transforms.ToTensor())
    indices_data = torch.load(f'{site_dir}/indices.pt')
    return Subset(full_train, indices_data['train_indices']), Subset(full_test, indices_data['test_indices'])

def train(num_sites):
    global_model = SimpleNet()
    loss_fn = nn.CrossEntropyLoss()
    
    # Federated learning rounds
    num_rounds = 5
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        
        site_models = []
        
        # Each site trains for one epoch starting from global model
        for site_id in range(num_sites):
            # Load only this site's data from disk
            train_data, test_data = load_site_data(site_id)
            
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=1000)
            
            # Clone global model to site
            site_model = SimpleNet()
            site_model.load_state_dict(global_model.state_dict())
            
            optimizer = optim.Adam(site_model.parameters(), lr=0.001)
            loss = train_one_epoch(site_model, train_loader, optimizer, loss_fn)
            
            accuracy = evaluate(site_model, test_loader, test_data)
            print(f"Site {site_id} - Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
            
            site_models.append(site_model)
        
        # Average the site models to update global model
        global_model.load_state_dict(average_weights(site_models))
        
        # Evaluate global model on full test set
        full_test = datasets.MNIST('/tmp/flare/dataset/', train=False, download=False, transform=transforms.ToTensor())
        full_test_loader = DataLoader(full_test, batch_size=1000)
        global_accuracy = evaluate(global_model, full_test_loader, full_test)
        print(f"\nGlobal Model Test Accuracy: {global_accuracy:.2f}%\n")

if __name__ == '__main__':
    num_sites = 3
    prepare_site_data(num_sites)
    train(num_sites)
