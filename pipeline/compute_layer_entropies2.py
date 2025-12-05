import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
import json
from tqdm import tqdm
from src.models.deeplabv3_mnv3 import get_empty_model, load_model
from pipeline.create_dataset import cityScapesDataset

class ActivationHistogram:
    def __init__(self, bins=256):
        self.bins = bins
        self.hist = None   # Will be a numpy array
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, tensor):
        """
        Update histogram with new activation tensor (torch.Tensor).
        Computes histogram on GPU then adds to running count.
        """
        # Flatten tensor for histogram computation
        tensor_flat = tensor.flatten()
        
        # Get min/max for this batch
        t_min = tensor_flat.min().item()
        t_max = tensor_flat.max().item()

        # Update global min/max
        self.min_val = min(self.min_val, t_min)
        self.max_val = max(self.max_val, t_max)

        # Compute histogram on GPU (torch.histc requires 1D tensor)
        h = torch.histc(tensor_flat, bins=self.bins, min=t_min, max=t_max)
        h = h.cpu().numpy()

        if self.hist is None:
            self.hist = h
        else:
            self.hist += h

    def finalize_entropy(self):
        """Convert histogram to probability distribution → Shannon entropy."""
        if self.hist is None:
            return None

        p = self.hist.astype(np.float64)
        p = p + 1e-12
        p /= p.sum()

        entropy = -np.sum(p * np.log2(p))
        return float(entropy)


# ==========================================================
# 2. Hook – capture activations from all leaf modules
# ==========================================================

activation_histograms = {}
layer_names = {}

def hook(module, inp, out):
    layer_name = layer_names[module]
    if isinstance(out, torch.Tensor):
        activation_histograms[layer_name].update(out.detach())


if __name__ == "__main__":
    # load pre-trained full-precision model
    print("Loading fine-tuned model...")
    model_checkpoint = "./models/finetuned_model_last_epoch.pth"  # TODO: CHANGE TO FINETUNED MODEL 
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    model = get_empty_model(num_classes=19)
    model = load_model(model, model_checkpoint, device=device)
    model = model.to(device)
    model.eval()

    # Create and register forward hooks to capture layer activations
    # Only register hooks on leaf modules (modules with no children)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            layer_names[module] = name
            activation_histograms[name] = ActivationHistogram(bins=256)
            module.register_forward_hook(hook)

    # Read config to get settings for data loading
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)
    batch_size = config["fp"]['training']['batch_size']
    transforms = config["fp"]['training']['val_transforms'] # no transformations

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

    for layer_name, hist in activation_histograms.items():
        entropy = hist.finalize_entropy()
        if entropy is not None:
            layer_entropies[layer_name] = entropy
            print(f"Layer {layer_name}: entropy = {entropy:.4f} bits")

    # Save results
    os.makedirs("results", exist_ok=True)
    entropies_output_file = "results/fp32_model_layer_entropies.json"
    with open(entropies_output_file, "w") as f:
        json.dump(layer_entropies, f, indent=2)
    print(f"\nEntropies saved to {entropies_output_file}")

    # Print results 
    print(f"\n{'='*50}")
    print("PER-LAYER ENTROPIES")
    print(f"{'='*50}")
    sorted_layers = sorted(layer_entropies.items(), key=lambda x: x[1], reverse=True)
    for layer, entropy in sorted_layers:
        print(f"{layer:50s}  entropy = {entropy:.4f} bits")
    print(f"\nTotal layers analyzed: {len(layer_entropies)}")
    
