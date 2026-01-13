# Federated learning simulation with nvflare collab api
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from nvflare.fox import fox
from nvflare.fox.sim import SimEnv
from nvflare.fox.sys.recipe import FoxRecipe

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))

@fox.collab
def train_site(global_state_dict):
    full_train = datasets.MNIST('/tmp/flare/dataset/', train=True, download=False, transform=transforms.ToTensor())
    full_test = datasets.MNIST('/tmp/flare/dataset/', train=False, download=False, transform=transforms.ToTensor())
    indices = torch.load(f'/tmp/mnist_experiment/{fox.site_name}/indices.pt')
    train_loader = DataLoader(Subset(full_train, indices['train_indices']), batch_size=64, shuffle=True)
    test_loader = DataLoader(Subset(full_test, indices['test_indices']), batch_size=1000)
    
    model = SimpleNet()
    model.load_state_dict(global_state_dict)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(data), target)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        correct = sum((model(data).argmax(1) == target).sum().item() for data, target in test_loader)
    accuracy = 100 * correct / len(indices['test_indices'])
    
    print(f"{fox.site_name} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    return model.state_dict()

@fox.algo
def federated_training():
    num_rounds = 5
    
    global_model = SimpleNet()
    
    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")
        
        site_state_dicts = [r for _,r in fox.clients.train_site(global_model.state_dict())] 
        global_model.load_state_dict({k: torch.stack([sd[k] for sd in site_state_dicts]).mean(0) for k in site_state_dicts[0]})
        
        # Evaluate global model
        test_loader = DataLoader(datasets.MNIST('/tmp/flare/dataset/', train=False, transform=transforms.ToTensor()), batch_size=1000)
        with torch.no_grad():
            correct = sum((global_model(d).argmax(1) == t).sum().item() for d, t in test_loader)
        print(f"Global Accuracy: {100 * correct / 10000:.2f}%")

if __name__ == '__main__':
    recipe = FoxRecipe(job_name="fox training", min_clients=3)
    run = recipe.execute(SimEnv(num_clients=3))

    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
