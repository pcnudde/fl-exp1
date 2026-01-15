# Simple MNIST training script
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))

def train():
    train_loader = DataLoader(datasets.MNIST('/tmp/flare/dataset/', train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('/tmp/flare/dataset/', train=False, download=True, transform=transforms.ToTensor()), batch_size=1000)
    
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(data), target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/5, Loss: {loss.item():.4f}')
    
    with torch.no_grad():
        correct = sum((model(data).argmax(1) == target).sum().item() for data, target in test_loader)
    print(f'Test Accuracy: {100 * correct / 10000:.2f}%')

if __name__ == '__main__':
    train()
