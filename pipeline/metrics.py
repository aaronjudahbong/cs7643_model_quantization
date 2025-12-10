import torch
import time
import json
import numpy as np
from typing import Tuple, Dict, Optional, Union

def calculate_miou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 19, ignore_index: int = 255) -> float:
    """
    Calculate mean Intersection over Union (mIoU) for semantic segmentation.
    
    Args:
        predictions: Tensor of shape [batch, H, W] with predicted class indices
        targets: Tensor of shape [batch, H, W] with ground truth class indices
        num_classes: Number of classes (excluding ignore_index)
        ignore_index: Class index to ignore in calculation
    
    Returns:
        (mIoU score (float), IoU (list))
    """
    # Create mask for valid pixels
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    mask = (target_flat != ignore_index)
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    # Calculate IoU for each class
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        # Skip classes that don't exist in ground truth or predictions
        if union == 0:
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    return (np.mean(ious), ious) if len(ious) > 0 else (0.0, [0.0])


def calculate_model_size(model: torch.nn.Module, quantization_bits: int = None) -> float:
    """
    Calculate model memory size in MB from model's parameters and buffers. Works on both 
    non-quantized and quantized models, provided the quantized model is converted using 
    quantize_fx.convert_fx.
    
    Args:
        model: PyTorch model
        quantization_bits: If provided, calculate theoretical size assuming parameters
                          are stored at this bit-width (e.g., 4 for int4). If None, 
                          calculates actual storage size.
    
    Returns:
        model_size_mb: In-memory size in MB
    """
    # Calculate size from model parameters
    if quantization_bits is not None:
        bytes_per_element = quantization_bits / 8.0
        param_size = sum(p.numel() * bytes_per_element for p in model.parameters())
    else:
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Torch buffers are not quantized, use actual size
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1000 ** 2)
    
    return model_size_mb

def calculate_model_size_mixed_precision(model: torch.nn.Module, quantization_bits: Optional[int] = None, layer_bit_depths: Optional[Union[Dict[str, int], str]] = None) -> float:
    """
    Calculate model memory size in MB from model's parameters and buffers. Works on both 
    non-quantized and quantized models (including models converted using quantize_fx.convert_fx).
    
    For quantized models after convert_fx, if model.parameters() is empty, this function
    uses state_dict() to access quantized tensors and calculate theoretical size based on
    the assigned bit depths.
    
    Args:
        model: PyTorch model (can be quantized or non-quantized)
        quantization_bits: If provided, calculate theoretical size assuming parameters
                          are stored at this bit-width (e.g., 4 for int4). Used for uniform
                          quantization. If None, calculates actual storage size.
        layer_bit_depths: For mixed-precision models, either:
                         - Dict mapping layer/module names to bit depths (8, 6, or 4)
                         - Path to JSON file containing layer bit depths mapping
                         If provided, overrides quantization_bits and uses per-layer bit depths.
                         Module names should match the original model (before quantization).
    
    Returns:
        model_size_mb: In-memory size in MB
    """
    # Load layer bit depths if path provided
    if isinstance(layer_bit_depths, str):
        with open(layer_bit_depths, 'r') as f:
            layer_bit_depths = json.load(f)
    
    # Calculate size from model parameters
    if layer_bit_depths is not None:
        # Mixed-precision: use per-layer bit depths
        # For quantized models after convert_fx, parameters might not be accessible via model.parameters()
        # So we use state_dict() which contains all quantized tensors
        
        # First, try the standard approach (works for non-quantized or prepared models)
        param_list = list(model.parameters())
        
        if len(param_list) == 0:
            # Quantized model: use state_dict() approach
            param_size = 0
            matched_params = 0
            unmatched_params = 0
            
            state_dict = model.state_dict()
            
            # Create a mapping from parameter names to module names
            # Parameter names are like "backbone.0.0.weight" -> module name is "backbone.0.0"
            for param_name, tensor in state_dict.items():
                # Skip buffers (they're not quantized, handled separately)
                if 'running_mean' in param_name or 'running_var' in param_name or 'num_batches_tracked' in param_name:
                    continue
                
                # Extract module name from parameter name
                # Parameter names are like "backbone.0.0.weight" -> module name is "backbone.0.0"
                # layer_bit_depths has module names without .weight, .bias, .scale, .zero_point, etc.
                if '.' in param_name:
                    # Remove common parameter suffixes to get module name
                    module_name = param_name
                    # Remove suffixes in order of specificity (longer first)
                    suffixes_to_remove = ['.zero_point', '.scale', '.bias', '.weight', '_packed_params']
                    for suffix in suffixes_to_remove:
                        if module_name.endswith(suffix):
                            module_name = module_name[:-len(suffix)]
                            break
                    
                    # Use exact match only to avoid confusion with multiple child modules
                    # (e.g., backbone.16.1 vs backbone.16.2 should not both match backbone.16)
                    if module_name in layer_bit_depths:
                        bit_depth = layer_bit_depths[module_name]
                        bytes_per_element = bit_depth / 8.0
                        param_size += tensor.numel() * bytes_per_element
                        matched_params += 1
                    else:
                        # Use 8-bit as fallback for unmatched parameters
                        # Don't do partial matching to avoid incorrect matches when multiple
                        # child modules exist (e.g., backbone.16.1, backbone.16.2, etc.)
                        bytes_per_element = 1.0
                        param_size += tensor.numel() * bytes_per_element
                        unmatched_params += 1
                else:
                    # Root level parameter, use 8-bit
                    bytes_per_element = 1.0
                    param_size += tensor.numel() * bytes_per_element
                    unmatched_params += 1
            
            if unmatched_params > 0:
                print(f"Warning: {unmatched_params} parameters not found in layer_bit_depths, using int8 size")
        else:
            # Standard approach: iterate through modules
            param_size = 0
            unmatched_modules = []
            
            # Iterate through modules (same as in compute_layer_entropies.py and mixed_precision.py)
            for module_name, module in model.named_modules():
                if module_name in layer_bit_depths:
                    bit_depth = layer_bit_depths[module_name]
                    bytes_per_element = bit_depth / 8.0
                    # Get all parameters for this module
                    for param in module.parameters(recurse=False):  # recurse=False to only get direct params
                        param_size += param.numel() * bytes_per_element
                else:
                    # Track unmatched modules for debugging
                    # Only count if module has parameters (to avoid noise from modules without params)
                    if sum(1 for _ in module.parameters(recurse=False)) > 0:
                        unmatched_modules.append(module_name)
                        # Use int8 as fallback
                        bytes_per_element = 1.0 
                        for param in module.parameters(recurse=False):
                            param_size += param.numel() * bytes_per_element
            
            if unmatched_modules:
                print(f"Warning: {len(unmatched_modules)} modules with parameters not found in layer_bit_depths, using int8 size")
    elif quantization_bits is not None:
        # Uniform quantization
        bytes_per_element = quantization_bits / 8.0
        param_list = list(model.parameters())
        if len(param_list) == 0:
            # Quantized model: use state_dict()
            param_size = 0
            state_dict = model.state_dict()
            for param_name, tensor in state_dict.items():
                # Skip buffers
                if 'running_mean' in param_name or 'running_var' in param_name or 'num_batches_tracked' in param_name:
                    continue
                param_size += tensor.numel() * bytes_per_element
        else:
            param_size = sum(p.numel() * bytes_per_element for p in param_list)
    else:
        # No quantization: use actual storage size
        param_list = list(model.parameters())
        if len(param_list) == 0:
            # Quantized model: use state_dict() with actual element sizes
            param_size = 0
            state_dict = model.state_dict()
            for param_name, tensor in state_dict.items():
                # Skip buffers
                if 'running_mean' in param_name or 'running_var' in param_name or 'num_batches_tracked' in param_name:
                    continue
                param_size += tensor.numel() * tensor.element_size()
        else:
            param_size = sum(p.numel() * p.element_size() for p in param_list)
    
    # Torch buffers are not quantized, use actual size
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1000 ** 2)
    
    return model_size_mb

def measure_inference_latency(model: torch.nn.Module, input_shape: Tuple[int, int, int], 
                             device: str, num_warmup: int = 10, num_runs: int = 100) -> dict:
    """
    Measure inference latency of the model.
    
    Args:
        model: PyTorch model in evaluation mode
        input_shape: Shape of input tensor (C, H, W)
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        num_warmup: Number of warmup runs
        num_runs: Number of runs of inference to average over
    
    Returns:
        Dictionary with latency statistics (mean, std, median) in milliseconds
    """
    # Set model to evaluation mode
    model.eval()

    # Create random input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Synchronize if using GPU
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    # Measure latency for random inputs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
            
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate latency statistics
    latency_stats = {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'median_ms': np.median(latencies)
    }

    return latency_stats