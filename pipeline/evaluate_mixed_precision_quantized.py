import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader

from src.models.deeplabv3_mnv3 import get_empty_model, load_model
from src.quantization.mixed_precision import build_qconfig_per_layer
from pipeline.create_dataset import cityScapesDataset

if __name__ == "__main__":
    print("--- Evaluating Quantized Mixed-Precision Model ---")
    
    # Device setup - quantized models don't work on MPS, use CPU or CUDA
    # MPS doesn't support quantized operations, so we need to use CPU
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'  # Use CPU instead of MPS for quantized models
    print(f"Using Device: {device} (quantized models don't support MPS)")
    
    # Load config
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)
    qat_config = config["qat"]
    
    # Load layer bit depths
    bit_depth_file = "results/mixed_precision_layer_bit_depths.json"
    print(f"Loading layer bit depths from: {bit_depth_file}")
    with open(bit_depth_file, "r") as f:
        layer_bit_depths = json.load(f)
    print(f"Loaded bit depth assignments for {len(layer_bit_depths)} layers")
    
    # Load base FP32 model
    checkpoint_path = "models/finetuned_model_last_epoch.pth"
    print(f"Loading base model from: {checkpoint_path}")
    model = get_empty_model(num_classes=19)
    model = load_model(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()
    
    # Build QConfigMapping (same as in mixed_precision.py)
    print("Building per-layer QConfigMapping...")
    qconfig_mapping = tq.QConfigMapping()
    # set global default qconfig (8-bit)
    default_qconfig = build_qconfig_per_layer(8, qat_config) 
    qconfig_mapping.set_global(default_qconfig)
    
    # Get model module dict to pass module objects
    model_module_dict = {name: module for name, module in model.named_modules()}
    
    # set per-layer qconfigs by overriding global default 
    for layer_name, bit_depth in layer_bit_depths.items():
        module = model_module_dict.get(layer_name)
        if module is None:
            print(f"[Warning] Could not find module: {layer_name}, skipping QConfig assignment")
            continue
        qconfig = build_qconfig_per_layer(bit_depth, qat_config, module)
        qconfig_mapping.set_module_name(layer_name, qconfig)
    
    # Skip ASPP
    if qat_config.get("skip_aspp", False):
        qconfig_mapping.set_module_name("classifier.0", None)
    
    # Prepare sample input for quantization
    cal_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_label_path = "data/gtFine_trainId/gtFine/train"
    train_transforms = qat_config['training']['train_transforms']
    cal_dataset = cityScapesDataset(cal_img_path, train_label_path, train_transforms)
    cal_loader = DataLoader(cal_dataset, batch_size=2, shuffle=True, drop_last=True)
    
    example_inputs, _ = next(iter(cal_loader))
    example_inputs = example_inputs.to(device)
    
    # Prepare and convert model to get quantized structure
    print("Preparing model for quantization...")
    prepared_model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, (example_inputs,))
    prepared_model = prepared_model.cpu()
    quantized_model = quantize_fx.convert_fx(prepared_model.eval())
    
    # Load quantized state_dict
    quantized_model_path = "models/mixed_precision_model.pth"
    print(f"Loading quantized model weights from: {quantized_model_path}")
    state_dict = torch.load(quantized_model_path, map_location='cpu')
    quantized_model.load_state_dict(state_dict)
    quantized_model = quantized_model.to(device)
    quantized_model.eval()
    
    # Setup validation dataset
    val_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/val"
    val_label_path = "data/gtFine_trainId/gtFine/val"
    val_transforms = {'crop': False, 'resize': False, 'flip': False}
    
    val_dataset = cityScapesDataset(val_img_path, val_label_path, val_transforms)
    # pin_memory only helps with CUDA, not CPU
    pin_memory = (device == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=pin_memory)
    
    num_classes = 19
    ignore_index = 255
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    # --------------------------------------------------
    # Inference loop (streaming, no prediction storage)
    # --------------------------------------------------
    print(f"\nRunning inference on validation set...")

    quantized_model.eval()
    with torch.no_grad():
        for i, (image, labels) in enumerate(val_loader):

            image = image.to(device, non_blocking=True)

            # forward pass
            out = quantized_model(image)
            if isinstance(out, dict):
                out = out["out"]

            preds = out.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            # Flatten per batch
            preds = preds.reshape(-1)
            targets = targets.reshape(-1)

            # Mask ignore index
            mask = targets != ignore_index
            preds = preds[mask]
            targets = targets[mask]

            # Update confusion matrix
            # bincount trick: (target * num_classes + pred)
            confusion += np.bincount(
                targets * num_classes + preds,
                minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(val_loader)}")

    # --------------------------------------------------
    # Compute IoU + mIoU
    # --------------------------------------------------
    print(f"\nCalculating mIoU...")

    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = intersection / np.maximum(union, 1)
    miou = np.nanmean(iou)

    print(f"\nmIoU: {miou:.4f}")
    
    # Print per-class IoUs
    print("\nPer-Class IoU:")
    for class_idx, iou_val in enumerate(iou):
        print(f"  Class {class_idx}: {iou_val:.4f}")


