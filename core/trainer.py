import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional

class AuraTrainer:
    """
    Device-agnostic trainer for AuraPrune Studio.
    Handles training loops, thermal lambda scheduling, and sparsity-aware loss.
    """
    def __init__(self, model: nn.Module, base_lambda: float = 1e-3, lr: float = 1e-3, device: Optional[torch.device] = None) -> None:
        # Device Agnostic setup (CUDA, MPS, or CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.base_lambda = base_lambda
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f" [AURA] Trainer initialized on device: {self.device}")

    def thermal_lambda_scheduler(self, epoch: int, start_epoch: int = 5, ramp_duration: int = 10) -> float:
        """
        Ramps up the sparsity penalty lambda after the initial warm-up period.
        """
        if epoch <= start_epoch:
            return 0.0
        
        # Gradual linear ramp-up
        ramp_factor = min(1.0, (epoch - start_epoch) / ramp_duration)
        return self.base_lambda * ramp_factor

    def calculate_sparsity_loss(self) -> torch.Tensor:
        """
        Calculates the L1 norm of all sigmoid gate values to encourage sparsity.
        """
        sparsity_loss = torch.tensor(0.0, device=self.device)
        for module in self.model.modules():
            if hasattr(module, 'gate_scores'):
                gates = torch.sigmoid(module.gate_scores)
                sparsity_loss += torch.sum(torch.abs(gates))
        return sparsity_loss

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """
        Executes one training epoch with thermal lambda scheduling.
        """
        self.model.train()
        running_loss = 0.0
        
        # Calculate active lambda for this epoch
        current_lambda = self.thermal_lambda_scheduler(epoch)
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Total Loss = CrossEntropy + λ * L1(gates)
            ce_loss = self.criterion(outputs, targets)
            sparsity_loss = self.calculate_sparsity_loss()
            total_loss = ce_loss + current_lambda * sparsity_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            
        return running_loss / len(train_loader), current_lambda

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluates accuracy and sparsity level on the test set.
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                    
        accuracy = 100. * correct / total
        
        # Calculate sparsity via separate utility if preferred, or inline
        from core.utils import calculate_sparsity_level
        sparsity = calculate_sparsity_level(self.model)
        
        return accuracy, sparsity
