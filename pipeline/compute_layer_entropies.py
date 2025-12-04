import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import yaml
import json
from tqdm import tqdm
from src.models.deeplabv3_mnv3 import get_empty_model, load_model
from pipeline.create_dataset import cityScapesDataset

def compute_entropy(values: np.ndarray, num_bins: int = 256) -> float: # TODO: HOW MANY BINS? 
    """
    Compute Shannon entropy from activation values.
    
    Args:
        values: 1D array of activations per layer
        num_bins: Number of bins for histogram
    
    Returns:
        entropy: Shannon entropy
    """
    # convert to probability distribution
    hist, bin_edges = np.histogram(values, bins=num_bins, density=False) # density=False to get counts then normalize to probabilities
    p = hist.astype(np.float64) + 1e-12  # avoid log(0)
    p = p / p.sum()  # normalize

    # calculate entropy
    entropy = -np.sum(p * np.log2(p))
    return entropy

def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
    """
    Forward hook function to capture layer activations
    
    Args:
        module: PyTorch module
        input: Input layer tensor
        output: Output layer tensor

    Returns:
        None
    """
    layer_name = module_to_name.get(module, str(module))
    
    # only capture activations from intermediate layers - tensors
    if isinstance(output, torch.Tensor):
        act = output.detach().cpu().flatten().numpy()
        
        # add to activation buffers
        if layer_name not in activation_buffers:
            activation_buffers[layer_name] = []
        activation_buffers[layer_name].append(act)

if __name__ == "__main__":
    # load pre-trained full-precision model
    print("Loading baseline model...")
    model_checkpoint = "./models/baseline_init_model.pth"  # TODO: CHANGE TO FINETUNED MODEL 
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    model = get_empty_model(num_classes=19)
    model = load_model(model, model_checkpoint, device=device)
    model = model.to(device)
    model.eval()

    # Create and register forward hooks to capture layer activations
    activation_buffers = {}
    module_to_name = {module: name for name, module in model.named_modules()}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            module.register_forward_hook(hook) # capture only leaf modules

    # Read config to get settings for data loading
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)
    batch_size = config["fp"]['training']['batch_size']
    transforms = config["fp"]['training']['train_transforms'] # simulate training conditions

    # Load data 
    print("Loading Cityscapes training dataset...")
    train_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_label_folder = "./data/gtFine_trainId/gtFine/train"
    training_dataset = cityScapesDataset(train_image_folder, train_label_folder, transforms)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) # same as fine_tuning.py

    # Collect activations through a single forward pass (match fine_tuning.py)
    print("Collecting activations (one full training epoch)...")
    print(f"Batch size: {batch_size}, Total batches: {len(training_loader)}")

    with torch.no_grad():
        for images, _ in tqdm(training_loader, desc="Collecting activations"):
            images = images.to(device, non_blocking=True)
            
            # hooks are automatically called on intermediate layers with forward pass
            _ = model(images)

    # Compute per-layer entropies on captured activations
    print("\nComputing per-layer entropies...")
    layer_entropies = {}

    for layer_name, activations in activation_buffers.items():
        if len(activations) == 0:
            continue

        activations_concatenated = np.concatenate(activations, axis=0) # concatenate all activations for each layer into one 1D array
        entropy = compute_entropy(activations_concatenated)
        layer_entropies[layer_name] = entropy


    # Save results: entropies and activations
    entropies_output_file = "results/fp32_model_layer_entropies.json"
    with open(entropies_output_file, "w") as f:
        json.dump(layer_entropies, f, indent=2)
    print(f"\nEntropies saved to {entropies_output_file}")

    activations_output_file = "results/fp32_model_layer_activations.json"
    with open(activations_output_file, "w") as f:
        json.dump(activation_buffers, f, indent=2)
    print(f"\nActivations saved to {activations_output_file}")

    # Print results 
    print(f"\n{'='*50}")
    print("PER-LAYER ENTROPIES")
    print(f"{'='*50}")
    sorted_layers = sorted(layer_entropies.items(), key=lambda x: x[1], reverse=True)
    for layer, entropy in sorted_layers:
        print(f"{layer:50s}  entropy = {entropy:.4f} bits")
    print(f"\nTotal layers analyzed: {len(layer_entropies)}")
    
