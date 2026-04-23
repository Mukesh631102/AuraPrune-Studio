import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math

class PrunableLinear(nn.Module):
    """
    1. Custom Layer: PrunableLinear
    Uses learnable gate_scores to determine active connections.
    """
    def __init__(self, in_features, out_features, noise_std=0.01):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.noise_std = noise_std
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate scores tensor with the same shape as weights
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.pruned_mask = None
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize gate scores such that sigmoid outputs values around 0.5-0.7
        nn.init.uniform_(self.gate_scores, 0.0, 1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        # Apply Sigmoid transformation to create gates between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # 3. Novelty Feature - Dynamic Thresholding
        # Adapts based on the layer's variance instead of a fixed threshold
        gate_mean = gates.mean()
        gate_std = gates.std() + 1e-8 # Add epsilon for stability
        
        # Threshold scales with variance, effectively capturing "weak" connections
        dynamic_threshold = gate_mean - 0.5 * gate_std
        active_mask = (gates > dynamic_threshold).float()
        
        # Cache pruned mask for the backward pass (Pruning Recovery)
        self.pruned_mask = (active_mask == 0.0)
        
        # Element-wise multiplication to apply the sparsity mask
        effective_gates = gates * active_mask
        effective_weights = self.weight * effective_gates
        
        return F.linear(x, effective_weights, self.bias)


class SelfPruningNet(nn.Module):
    """
    Self-Pruning Network built for the CIFAR-10 Dataset.
    Combines standard convolutions with PrunableLinear layers.
    """
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        # Standard feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Prunable fully connected layers
        # Input size handling 64 channels * 8 * 8 spatial dimensions
        self.classifier = nn.Sequential(
            PrunableLinear(64 * 8 * 8, 512),
            nn.ReLU(),
            PrunableLinear(512, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def calculate_sparsity_loss(model):
    """
    2. Sparsity Logic: Compute the L1-norm Regularization of the gate scores.
    """
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss += torch.sum(torch.abs(gates))
    return sparsity_loss


def inject_gradient_noise(model):
    """
    4. Novelty Feature - Pruning Recovery
    Adds a small gradient noise factor to pruned gates. 
    Allows the model to re-explore pruned pathways during backpropagation.
    """
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            if module.gate_scores.grad is not None and module.pruned_mask is not None:
                # Generate noise and apply only to pruned indices
                noise = torch.randn_like(module.gate_scores.grad) * module.noise_std
                module.gate_scores.grad += noise * module.pruned_mask.float()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # Setup CIFAR-10 Dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training configurations
    epochs = 5
    lambda_sparsity = 1e-4  # Weight scalar for sparsity regularization
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            cls_loss = criterion(outputs, labels)
            
            # Sparsity Logic implementation
            # Total Loss = ClassificationLoss + lambda * SparsityLoss
            sparsity_loss = calculate_sparsity_loss(model)
            total_loss = cls_loss + lambda_sparsity * sparsity_loss
            
            # Backward pass
            total_loss.backward()
            
            # Pruning Recovery implementation
            # Intercepts gradients right before taking the optimization step
            inject_gradient_noise(model)
            
            optimizer.step()
            
            running_loss += total_loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training.")

if __name__ == '__main__':
    train()
