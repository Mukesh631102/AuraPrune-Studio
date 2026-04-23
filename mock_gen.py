import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_mock_results():
    results = [
        {"lambda": 0.0001, "final_accuracy": 45.2, "final_sparsity": 8.5},
        {"lambda": 0.001, "final_accuracy": 41.8, "final_sparsity": 32.1},
        {"lambda": 0.01, "final_accuracy": 30.5, "final_sparsity": 78.4}
    ]
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Mock results saved to results.json")

    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    # Generate a dummy gate distribution plot
    gate_values = np.concatenate([
        np.random.normal(0.1, 0.05, 1000), # Pruned
        np.random.normal(0.9, 0.05, 3000), # Active
        np.random.uniform(0, 1, 500)      # Transitioning
    ])
    gate_values = np.clip(gate_values, 0, 1)
            
    plt.figure(figsize=(10, 6))
    plt.hist(gate_values, bins=50, color='#00F0FF', edgecolor='black', alpha=0.7)
    plt.title('Final Gate Value Distribution (Bimodal Pruning)', color='white')
    plt.xlabel('Gate Value (Sigmoid Score)', color='white')
    plt.ylabel('Frequency', color='white')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.gcf().set_facecolor('#05050A')
    plt.gca().set_facecolor('#05050A')
    plt.tick_params(colors='white')
    
    plt.savefig('assets/gate_distribution.png')
    print("Mock plot saved to assets/gate_distribution.png")

if __name__ == "__main__":
    generate_mock_results()
