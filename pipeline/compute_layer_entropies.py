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
        self.hist = None # np.ndarray
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update_minmax(self, tensor: torch.Tensor):
        """
        First pass: Update global min/max only (no histogram computation).
        """
        tensor_flat = tensor.detach().flatten()
        t_min = tensor_flat.min().item()
        t_max = tensor_flat.max().item()
        self.min_val = min(self.min_val, t_min)
        self.max_val = max(self.max_val, t_max)

    def update(self, tensor: torch.Tensor):
        """
        Second pass: Update histogram with new activation tensor using global min/max.

        Args:
            tensor: Activation tensor to update histogram with

        Returns:
            None
        """
        tensor_flat = tensor.detach().flatten()
        
        # Compute histogram using global min/max for consistent binning
        h = torch.histc(tensor_flat, bins=self.bins, min=self.min_val, max=self.max_val)
        h = h.cpu().numpy()

        if self.hist is None:
            self.hist = h
        else:
            self.hist += h

    def finalize_entropy(self):
        """
        Convert histogram to probability distribution and calculate Shannon entropy.
        """
        if self.hist is None:
            return None

        p = self.hist.astype(np.float64)
        p = p + 1e-12 # avoid log(0)
        p /= p.sum()

        entropy = -np.sum(p * np.log2(p))
        return float(entropy)

activation_histograms = {}
layer_names = {}

def hook_minmax(module: nn.Module, _input: torch.Tensor, output: torch.Tensor):
    """First pass: collect global min/max only."""
    layer_name = layer_names[module]
    if isinstance(output, torch.Tensor):
        activation_histograms[layer_name].update_minmax(output.detach())

def hook(module: nn.Module, _input: torch.Tensor, output: torch.Tensor):
    """
    Second pass: compute histogram with global min/max.

    Args:
        module: Module to capture activations from
        _input: Input tensor (unused)
        output: Output tensor to capture activations from
    """
    layer_name = layer_names[module]
    if isinstance(output, torch.Tensor):
        activation_histograms[layer_name].update(output.detach())


if __name__ == "__main__":
    # load pre-trained full-precision model
    print("Loading fine-tuned model...")
    model_checkpoint = "./models/finetuned_model_last_epoch.pth"
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    model = get_empty_model(num_classes=19)
    model = load_model(model, model_checkpoint, device=device)
    model = model.to(device)
    model.eval()

    # Create and register forward hooks to capture layer activations on leaf modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # only leaf modules
            layer_names[module] = name
            activation_histograms[name] = ActivationHistogram(bins=256)

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

    # Collect activations through 1024 images only
    num_images = 1024
    num_batches = (num_images + batch_size - 1) // batch_size  # ceiling division
    print(f"Collecting activations from {num_images} images ({num_batches} batches)...")
    print(f"Batch size: {batch_size}")

    # First pass: collect global min/max
    print("First pass: Collecting global min/max...")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            module.register_forward_hook(hook_minmax)
    
    with torch.no_grad():
        image_count = 0
        for images, _ in tqdm(training_loader, desc="Pass 1: Min/Max"):
            images = images.to(device, non_blocking=True)
            _ = model(images)
            image_count += images.size(0)
            if image_count >= num_images:
                break
    
    # Remove first pass hooks and register second pass hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            # Remove old hooks (PyTorch doesn't have direct remove, so we'll re-register)
            module._forward_hooks.clear()
            module.register_forward_hook(hook)
    
    # Second pass: compute histograms with global min/max
    print("Second pass: Computing histograms with global min/max...")
    # Reset DataLoader to start from beginning
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    with torch.no_grad():
        image_count = 0
        for images, _ in tqdm(training_loader, desc="Pass 2: Histograms"):
            images = images.to(device, non_blocking=True)
            _ = model(images)
            image_count += images.size(0)
            if image_count >= num_images:
                break

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
    print(f"\nTotal layers analyzed: {len(layer_entropies)}")
    