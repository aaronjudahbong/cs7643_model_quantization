import torch
import time
import numpy as np
from typing import Tuple

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