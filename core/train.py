import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from core.model import AuraPruneNet
import os

def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def calculate_sparsity_loss(model):
    sparsity_loss = 0
    gate_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss += torch.sum(torch.abs(gates))
            gate_count += gates.numel()
    return sparsity_loss

def get_sparsity_level(model, threshold=0.01):
    total_gates = 0
    pruned_gates = 0
    with torch.no_state_dict():
        for module in model.modules():
            if hasattr(module, 'gate_scores'):
                gates = torch.sigmoid(module.gate_scores)
                total_gates += gates.numel()
                pruned_gates += torch.sum(gates < threshold).item()
    return (pruned_gates / total_gates) * 100 if total_gates > 0 else 0

def train(model, train_loader, optimizer, criterion, lambda_sparsity, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        ce_loss = criterion(outputs, targets)
        sparsity_loss = calculate_sparsity_loss(model)
        
        total_loss = ce_loss + lambda_sparsity * sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
    
    return running_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    accuracy = 100. * correct / total
    sparsity = get_sparsity_level(model)
    return accuracy, sparsity

if __name__ == "__main__":
    # This block can be used for testing individual runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuraPruneNet().to(device)
    train_loader, test_loader = get_dataloaders()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    lambda_val = 1e-3
    for epoch in range(1, 6): # Short run for demo
        loss = train(model, train_loader, optimizer, criterion, lambda_val, device)
        acc, spar = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Sparsity: {spar:.2f}%")
