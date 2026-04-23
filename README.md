# AuraPrune Studio 🌌

AuraPrune is a high-performance **Self-Pruning Neural Network** system designed for the CIFAR-10 dataset. It utilizes learnable sigmoid gates and an L1 regularization framework to autonomously identify and retract redundant weights during the training process, achieving a state-of-the-art balance between precision and architectural sparsity.

![Dashboard Preview](assets/gate_distribution.png)

## 🚀 Key Features

*   **Self-Pruning Architecture**: Custom `PrunableLinear` layers that learn their own connectivity via backpropagation.
*   **Thermal Lambda Scheduler**: A dynamic regularization engine that warms up feature learning for the first 5 epochs before gradually ramping up the pruning pressure.
*   **Anti-Gravity Dashboard**: A real-time HTML5/JS dashboard with 3D Plotly visualizations, live telemetry sync, and interactive weight-retraction animations.
*   **Device Agnostic Engine**: Built-in support for **CUDA** (NVIDIA), **MPS** (Apple Metal), and standard **CPU** training.
*   **Automated Experiments**: Integrated triple-run logic to explore the trade-offs between different regularization coefficients (λ).

## 🛠️ Project Structure

```text
├── core/
│   ├── model.py      # PrunableLinear and AuraPruneNet architecture
│   ├── trainer.py    # Device-agnostic trainer with Thermal Scheduler
│   ├── utils.py      # Sparsity telemetry and JSON export logic
│   └── train.py      # Dataset loading and basic training loop
├── assets/           # Visualization assets and gate distribution plots
├── dashboard.html    # Interactive Anti-Gravity Frontend
├── experiments.py    # Master script for running lambda trials
├── main.py           # FastAPI service for live model telemetry
├── REPORT.md         # Detailed mathematical intuition and results
└── results.json      # Live telemetry data feed for the UI
```

## ⚡ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Mukesh631102/AuraPrune-Studio.git
cd AuraPrune-Studio
pip install torch torchvision matplotlib plotly fastapi uvicorn
```

### 2. Run Experiments
Execute the triple-run experiment suite to generate model metrics:
```bash
python experiments.py
```

### 3. Launch Dashboard
Open `index.html` in your browser to view the live 3D network architecture and performance telemetry.

### 4. Interactive Development
Generate and run the project via the provided Jupyter Notebook builder:
```bash
python build_auraprune.py
```

## 🧠 Mathematical Intuition
The system adds a sparsity penalty to the standard Cross-Entropy loss:
**Total Loss = CrossEntropy + Lambda * Sum(sigmoid(gate_scores))**

As training progresses, the **Thermal Scheduler** increases Lambda, forcing weights with low contribution to classification accuracy to have their gate scores pushed toward negative infinity, effectively "pruning" them from the network.

---
*Developed for the Tredence Analytics AI Engineer Case Study.*
