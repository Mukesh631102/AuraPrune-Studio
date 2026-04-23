import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AuraPrune: Self-Pruning Neural Network\n",
                "\n",
                "This dynamic neural network uses custom `PrunableLinear` layers coupled with a Thermal Lambda Scheduler and Recovery Momentum to self-prune its less significant weights during training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.nn.functional as F\n",
                "import torch.optim as optim\n",
                "import torchvision\n",
                "import torchvision.transforms as transforms\n",
                "import math\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Modular Prunable Linear Layer\n",
                "A robust replacement for `nn.Linear` integrating learnable `gate_scores`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class PrunableLinear(nn.Module):\n",
                "    def __init__(self, in_features, out_features, noise_std=0.01):\n",
                "        super(PrunableLinear, self).__init__()\n",
                "        self.in_features = in_features\n",
                "        self.out_features = out_features\n",
                "        self.noise_std = noise_std\n",
                "        \n",
                "        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))\n",
                "        self.bias = nn.Parameter(torch.Tensor(out_features))\n",
                "        \n",
                "        # Learnable gate scores mapped to weight tensor dimensions\n",
                "        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))\n",
                "        self.pruned_mask = None\n",
                "        self.reset_parameters()\n",
                "        \n",
                "    def reset_parameters(self):\n",
                "        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))\n",
                "        # Distribute gate scores logically\n",
                "        nn.init.uniform_(self.gate_scores, 0.0, 1.0)\n",
                "        if self.bias is not None:\n",
                "            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)\n",
                "            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0\n",
                "            nn.init.uniform_(self.bias, -bound, bound)\n",
                "            \n",
                "    def forward(self, x):\n",
                "        # Active gated probabilities between 0 and 1\n",
                "        gates = torch.sigmoid(self.gate_scores)\n",
                "        \n",
                "        # Capture statistical distribution of gates\n",
                "        gate_mean = gates.mean()\n",
                "        gate_std = gates.std() + 1e-8\n",
                "        \n",
                "        # Dynamic local variance threshold\n",
                "        dynamic_threshold = gate_mean - 0.5 * gate_std\n",
                "        active_mask = (gates > dynamic_threshold).float()\n",
                "        \n",
                "        # Store cached dead weights for Recovery Momentum phase\n",
                "        self.pruned_mask = (active_mask == 0.0)\n",
                "        \n",
                "        # Apply sparsity computationally\n",
                "        effective_weights = self.weight * (gates * active_mask)\n",
                "        return F.linear(x, effective_weights, self.bias)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. AuraPrune Model Definition"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class AuraPruneNet(nn.Module):\n",
                "    def __init__(self):\n",
                "        super(AuraPruneNet, self).__init__()\n",
                "        self.features = nn.Sequential(\n",
                "            nn.Conv2d(3, 32, kernel_size=3, padding=1),\n",
                "            nn.ReLU(),\n",
                "            nn.MaxPool2d(2, 2),\n",
                "            nn.Conv2d(32, 64, kernel_size=3, padding=1),\n",
                "            nn.ReLU(),\n",
                "            nn.MaxPool2d(2, 2)\n",
                "        )\n",
                "        self.classifier = nn.Sequential(\n",
                "            PrunableLinear(64 * 8 * 8, 256),\n",
                "            nn.ReLU(),\n",
                "            PrunableLinear(256, 128),\n",
                "            nn.ReLU(),\n",
                "            PrunableLinear(128, 10)\n",
                "        )\n",
                "        \n",
                "    def forward(self, x):\n",
                "        x = self.features(x)\n",
                "        x = x.view(x.size(0), -1)\n",
                "        return self.classifier(x)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Thermal Lambda Scheduler and Recovery Momentum\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def get_thermal_lambda(epoch, max_epochs, lambda_max, tau=2.0):\n",
                "    \"\"\"Gradually scales Lambda up to avoid enforcing sparsity prematurely.\"\"\"\n",
                "    # Thermal formula: L_t = L_max * (1 - e^(-t / tau))\n",
                "    return lambda_max * (1 - math.exp(-epoch / tau))\n",
                "\n",
                "def calculate_vectorized_sparsity_loss(model):\n",
                "    \"\"\"Pre-computed autograd enabled L1 penalty\"\"\"\n",
                "    loss = 0.0\n",
                "    # Vectorized accumulation across layers\n",
                "    for module in model.modules():\n",
                "        if isinstance(module, PrunableLinear):\n",
                "            loss += torch.sum(torch.abs(torch.sigmoid(module.gate_scores)))\n",
                "    return loss\n",
                "\n",
                "def apply_recovery_momentum(model):\n",
                "    \"\"\"Injects random normal distributed noise specifically into 'dead' gate gradients.\"\"\"\n",
                "    for module in model.modules():\n",
                "        if isinstance(module, PrunableLinear):\n",
                "            if module.gate_scores.grad is not None and module.pruned_mask is not None:\n",
                "                noise = torch.randn_like(module.gate_scores.grad) * module.noise_std\n",
                "                # Injects directly into gradient buffer\n",
                "                module.gate_scores.grad += noise * module.pruned_mask.float()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Training Engine Setup (CIFAR-10)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_model(model, valloader, device):\n",
                "    model.eval()\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    with torch.no_grad():\n",
                "        for data in valloader:\n",
                "            images, labels = data\n",
                "            images, labels = images.to(device), labels.to(device)\n",
                "            outputs = model(images)\n",
                "            _, predicted = torch.max(outputs.data, 1)\n",
                "            total += labels.size(0)\n",
                "            correct += (predicted == labels).sum().item()\n",
                "    return 100 * correct / total\n",
                "\n",
                "def calculate_network_sparsity(model):\n",
                "    pruned = 0\n",
                "    total = 0\n",
                "    with torch.no_grad():\n",
                "        for module in model.modules():\n",
                "            if isinstance(module, PrunableLinear):\n",
                "                gates = torch.sigmoid(module.gate_scores)\n",
                "                gate_mean = gates.mean()\n",
                "                gate_std = gates.std() + 1e-8\n",
                "                dynamic_threshold = gate_mean - 0.5 * gate_std\n",
                "                active_mask = (gates > dynamic_threshold).float()\n",
                "                \n",
                "                total += active_mask.numel()\n",
                "                pruned += (active_mask == 0.0).sum().item()\n",
                "    return (pruned / total) * 100 if total > 0 else 0"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "try:\n",
                "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "    print(f\"Device Successfully Detected: {device}\")\n",
                "    if device.type == 'cuda':\n",
                "        print(f\"GPU Name: {torch.cuda.get_device_name(0)}\")\n",
                "except Exception as e:\n",
                "    print(f\"Warning: Error detecting GPU. Defaulting to CPU.\\n{str(e)}\")\n",
                "    device = torch.device(\"cpu\")\n",
                "\n",
                "transform = transforms.Compose([\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\n",
                "])\n",
                "\n",
                "try:\n",
                "    print(\"Ingesting CIFAR-10 Dataset...\")\n",
                "    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)\n",
                "    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)\n",
                "    \n",
                "    # Create a subset to complete mock training rapidly.\n",
                "    train_subset = torch.utils.data.Subset(trainset, range(0, 5000))\n",
                "    val_subset = torch.utils.data.Subset(valset, range(0, 1000))\n",
                "    \n",
                "    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)\n",
                "    valloader = torch.utils.data.DataLoader(val_subset, batch_size=128, shuffle=False)\n",
                "    print(\"Data Ingestion Phase Completed Successfully.\")\n",
                "except Exception as e:\n",
                "    print(f\"CRITICAL ERROR during Dataset Loading phase:\\n{str(e)}\")\n",
                "    raise"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lambda_profiles = {\n",
                "    \"Low\": 1e-5,\n",
                "    \"Medium\": 1e-3,\n",
                "    \"High\": 5e-2\n",
                "}\n",
                "epochs = 3\n",
                "results = []\n",
                "models = {}\n",
                "\n",
                "for profile, max_lambda in lambda_profiles.items():\n",
                "    print(f\"\\n--- Training with {profile} Lambda Profile ---\")\n",
                "    model = AuraPruneNet().to(device)\n",
                "    optimizer = optim.Adam(model.parameters(), lr=0.002)\n",
                "    criterion = nn.CrossEntropyLoss()\n",
                "\n",
                "    for epoch in range(epochs):\n",
                "        model.train()\n",
                "        current_lambda = get_thermal_lambda(epoch, epochs, max_lambda)\n",
                "        \n",
                "        for i, (inputs, labels) in enumerate(trainloader):\n",
                "            inputs, labels = inputs.to(device), labels.to(device)\n",
                "            optimizer.zero_grad()\n",
                "            \n",
                "            outputs = model(inputs)\n",
                "            cls_loss = criterion(outputs, labels)\n",
                "            \n",
                "            # Compute Sparsity via Thermal Lambda\n",
                "            sparsity_penalty = calculate_vectorized_sparsity_loss(model)\n",
                "            loss = cls_loss + (current_lambda * sparsity_penalty)\n",
                "            \n",
                "            loss.backward()\n",
                "            \n",
                "            # Recovery momentum intercepts here\n",
                "            apply_recovery_momentum(model)\n",
                "            \n",
                "            optimizer.step()\n",
                "\n",
                "    # Collect engine metrics mapping\n",
                "    test_acc = evaluate_model(model, valloader, device)\n",
                "    sparsity = calculate_network_sparsity(model)\n",
                "    results.append({\n",
                "        \"Lambda Profile\": profile,\n",
                "        \"Test Accuracy (%)\": round(test_acc, 2),\n",
                "        \"Sparsity Level (%)\": round(sparsity, 2)\n",
                "    })\n",
                "    models[profile] = model\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Final Outputs Analysis\n",
                "Generate the comparison dataframe output and Matplotlib Distribution visualization of Gate Scores."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Table Comparing Lambda Values\n",
                "df_results = pd.DataFrame(results)\n",
                "display(df_results)\n",
                "\n",
                "# 2. Gate Distribution Histogram\n",
                "target_model = models[\"Medium\"] # Sample the balanced version\n",
                "all_gates = []\n",
                "\n",
                "with torch.no_grad():\n",
                "    for module in target_model.modules():\n",
                "        if isinstance(module, PrunableLinear):\n",
                "            gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()\n",
                "            all_gates.extend(gates)\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.hist(all_gates, bins=50, color='cyan', alpha=0.7, edgecolor='white')\n",
                "plt.title('Gate Distribution (Anti-Gravity Sparsity Profiling)')\n",
                "plt.xlabel('Sigmoid Gate Output (0.0 to 1.0)')\n",
                "plt.ylabel('Parameter Count')\n",
                "plt.style.use('dark_background') # Match the Studio aesthetics globally\n",
                "plt.grid(color='white', alpha=0.1)\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'e:\Tredence Analytics\AuraPrune.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("AuraPrune.ipynb successfully generated!")
