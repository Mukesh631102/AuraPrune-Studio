import torch
import torch.nn as nn
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any
from core.model import AuraPruneNet
from core.trainer import AuraTrainer
from core.train import get_dataloaders
from core.utils import export_telemetry

def run_experiment(lambda_val: float, epochs: int = 15) -> Tuple[float, float, nn.Module]:
    """
    Executes a training experiment for a specific lambda coefficient.
    """
    print(f"\n>>> INITIATING EXPERIMENT | lambda = {lambda_val}")
    
    # Initialize Data and Model
    train_loader, test_loader = get_dataloaders()
    model = AuraPruneNet()
    trainer = AuraTrainer(model, base_lambda=lambda_val)
    
    final_acc = 0.0
    final_spar = 0.0
    
    for epoch in range(1, epochs + 1):
        loss, current_lambda = trainer.train_epoch(train_loader, epoch)
        acc, spar = trainer.evaluate(test_loader)
        
        # Real-time UI Export
        export_telemetry(epoch, acc, spar, current_lambda)
        
        final_acc = acc
        final_spar = spar
        
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Accuracy: {acc:.2f}% | Sparsity: {spar:.2f}%")
        
    return final_acc, final_spar, trainer.model

def generate_visualization(model: nn.Module, save_path: str = "assets/gate_distribution.png") -> None:
    """
    Generates a high-quality histogram of final gate values highlighting bimodal distribution.
    """
    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    gate_values = []
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'gate_scores'):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                gate_values.extend(gates)
                
    plt.figure(figsize=(12, 7))
    plt.hist(gate_values, bins=60, color='#00F0FF', edgecolor='#05050A', alpha=0.85)
    
    # Anti-Gravity Styling
    plt.title('Neural Architecture: Final Gate Distribution', fontsize=16, color='white', pad=20)
    plt.xlabel('Gate Retraction Level (Sigmoid Score)', fontsize=12, color='white')
    plt.ylabel('Weight Count', fontsize=12, color='white')
    
    plt.gcf().set_facecolor('#05050A')
    plt.gca().set_facecolor('#05050A')
    plt.gca().spines['bottom'].set_color('#8F95A3')
    plt.gca().spines['left'].set_color('#8F95A3')
    plt.gca().tick_params(colors='white')
    plt.grid(axis='y', linestyle='--', alpha=0.1)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" [VIZ] High-quality histogram saved to {save_path}")

def main() -> None:
    """
    Executes the triple-run experimental logic for the Case Study.
    """
    # Reset results file at start of full run
    if os.path.exists("results.json"):
        os.remove("results.json")
        
    lambdas: List[float] = [1e-4, 1e-3, 1e-2]
    best_model: Optional[nn.Module] = None
    max_acc: float = 0.0
    
    for l in lambdas:
        acc, spar, model = run_experiment(l, epochs=3) # Short epochs for case study demonstration
        
        if acc > max_acc:
            max_acc = acc
            best_model = model
            
    if best_model:
        generate_visualization(best_model)

if __name__ == "__main__":
    main()
