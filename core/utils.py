import torch
import torch.nn as nn
import json
import os
from typing import List, Dict, Any

def calculate_sparsity_level(model: nn.Module, threshold: float = 0.01) -> float:
    """
    Computes the percentage of weights dynamically pruned across the network.
    A connection is considered pruned if its gate value is below the threshold.
    """
    total_gates = 0
    pruned_gates = 0
    
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'gate_scores'):
                gates = torch.sigmoid(module.gate_scores)
                total_gates += gates.numel()
                pruned_gates += torch.sum(gates < threshold).item()
                
    return (pruned_gates / total_gates) * 100 if total_gates > 0 else 0.0

def export_telemetry(epoch: int, accuracy: float, sparsity: float, lambda_val: float, filename: str = "results.json") -> None:
    """
    Exports training metrics to a JSON file to drive the Anti-Gravity frontend visualizations.
    """
    data: List[Dict[str, Any]] = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = []
            
    # Record current snapshot
    data.append({
        "epoch": int(epoch),
        "accuracy": round(float(accuracy), 2),
        "sparsity_percentage": round(float(sparsity), 2),
        "lambda_value": float(lambda_val)
    })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f" [DATA] Sync complete: {filename}")
