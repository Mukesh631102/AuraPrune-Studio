import io
import hashlib
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from PIL import Image
import torchvision.transforms as transforms

# Importing the custom Self-Pruning Model architecture
from self_pruning_network import SelfPruningNet, PrunableLinear


app = FastAPI(
    title="Tredence Self-Pruning Service",
    description="High-performance async FastAPI service supporting the dynamic self-pruning neural network metrics.",
    version="1.0.0"
)


# ----------------- #
# Infrastructure    #
# ----------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model. (In production, load with `model.load_state_dict()`)
model = SelfPruningNet().to(device)
model.eval()  

# Transformation tailored for CIFAR-10 inference (32x32 pixels)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# 4. Basic In-memory cache for optimizing repeated requests
# Note: For multi-worker production deployments, replace this with Redis.
_prediction_cache: Dict[str, Any] = {}


# ----------------- #
# Pydantic Schemas  #
# ----------------- #

class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    sparsity_level: float
    agent_recommendation: str

class TelemetryResponse(BaseModel):
    global_sparsity: float
    layer_distributions: Dict[str, Dict[str, int]]


# ----------------- #
# Helper Functions  #
# ----------------- #

def calculate_dynamic_sparsity() -> float:
    """Computes the percentage of weights dynamically pruned across the network."""
    total_params = 0
    pruned_params = 0
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                # Mimic the dynamic local variance thresholding from training
                gate_mean = gates.mean()
                gate_std = gates.std() + 1e-8
                dynamic_threshold = gate_mean - 0.5 * gate_std
                
                active_mask = (gates > dynamic_threshold).float()
                
                total_params += active_mask.numel()
                pruned_params += (active_mask == 0.0).sum().item()
                
    if total_params == 0:
        return 0.0
    return (pruned_params / total_params) * 100.0


def run_local_agent(sparsity: float, confidence: float) -> str:
    """
    3. Novelty Feature - "Agentic Feedback"
    Rule-based local evaluating workflow logic. Evaluates sparsity against precision 
    and offers a tactical recommendation on lambda (λ) hyperparameter shifting. 
    """
    if confidence > 0.85 and sparsity < 30.0:
        return "ACTION_INCREASE_LAMBDA: Model identifies high confidence with low pruning severity. Safely increase lambda (λ) to promote optimization."
        
    elif confidence < 0.60 and sparsity > 65.0:
        return "ACTION_DECREASE_LAMBDA: Degradation logic triggered. Confidence corresponds poorly with high sparsity. Decrease lambda (λ) to enforce pruning recovery."
        
    elif sparsity > 85.0:
        return "ACTION_DECREASE_LAMBDA: Extreme hyper-sparsity levels detected. Warning: Informational plateau risk."
        
    return "ACTION_MAINTAIN: Model is securely balanced in a stable sparsity/accuracy plateau."


# ----------------- #
# API Endpoints     #
# ----------------- #

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    1. Inference Endpoint
    Handles image uploads asynchronously, executes the self-pruning forward pass, 
    evaluates confidence, and attaches localized ML-agent diagnostics.
    """
    image_bytes = await file.read()
    
    # Retrieve from in-memory cache directly if processed earlier
    file_hash = hashlib.md5(image_bytes).hexdigest()
    if file_hash in _prediction_cache:
        return _prediction_cache[file_hash]
        
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Passed object is not a valid image format.")
        
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        
    confidence_val = conf.item()
    class_idx = predicted.item()
    
    # Capture telemetry metrics inline
    sparsity_level = calculate_dynamic_sparsity()
    
    # Utilize Agent Logic workflow
    agent_advice = run_local_agent(sparsity_level, confidence_val)
    
    response_data = PredictResponse(
        class_id=class_idx,
        class_name=CLASSES[class_idx],
        confidence=confidence_val,
        sparsity_level=sparsity_level,
        agent_recommendation=agent_advice
    )
    
    # Store serialization
    _prediction_cache[file_hash] = response_data
    return response_data


@app.get("/stats", response_model=TelemetryResponse)
async def telemetry_endpoint():
    """
    2. Telemetry Endpoint
    Serves a unified payload defining absolute sparsity and mapping the explicit 
    histogram binning of the gate values in 10-point scale segments.
    """
    global_sparsity = calculate_dynamic_sparsity()
    distributions = {}
    
    with torch.no_grad():
        layer_index = 0
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                # Distribute the gates mathematically via [0-1] domain binning 
                hist = torch.histc(gates.cpu(), bins=10, min=0, max=1)
                
                layer_dist = {}
                for bin_idx, count in enumerate(hist.tolist()):
                    bucket_start = f"{bin_idx * 0.1:.1f}"
                    bucket_end = f"{(bin_idx + 1) * 0.1:.1f}"
                    layer_dist[f"{bucket_start}-{bucket_end}"] = int(count)
                    
                distributions[f"layer_{layer_index}_prunable_linear"] = layer_dist
                layer_index += 1
                
    return TelemetryResponse(
        global_sparsity=global_sparsity,
        layer_distributions=distributions
    )


if __name__ == "__main__":
    import uvicorn
    # Initiating Async ASGI loop
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
