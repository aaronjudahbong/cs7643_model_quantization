import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.deeplabv3_mnv3 import get_empty_model, load_model, save_model
from .quantization_utils import set_seed
from pipeline.create_dataset import cityScapesDataset
from pipeline.metrics import calculate_miou, calculate_model_size_mixed_precision
import matplotlib.pyplot as plt
from src.scripts.mappings import map_train_id_to_color
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

def build_qconfig_per_layer(bit_depth: int, config: dict, module: nn.Module = None) -> tq.QConfig:
    """
    Build a QConfig for each layer based on specified bit depth.
    
    Args:
        bit_depth: 8, 6, or 4
        config: Base config dict
        module: Optional module object used to check if it's Hardsigmoid
    
    Returns:
        QConfig for the layer
    """
    # Define quantization ranges based on bit depth
    if bit_depth == 8:
        weight_quant_min, weight_quant_max = -128, 127
        act_quant_min, act_quant_max = 0, 127
    elif bit_depth == 6:
        weight_quant_min, weight_quant_max = -32, 31
        act_quant_min, act_quant_max = 0, 31
    elif bit_depth == 4:
        weight_quant_min, weight_quant_max = -8, 7
        act_quant_min, act_quant_max = 0, 7
    else:
        raise ValueError(f"Bit depth {bit_depth} not supported - must be 8, 6, or 4")
    
    # Build QConfig for weights
    weight_dtype = torch.qint8 if config["weights"]["dtype"] == "qint8" else torch.quint8 # torch doesn't support lower bit depths, simulate based on ranges above
    weights_granularity = config["weights"]["granularity"]
    if weights_granularity == "per_channel":
        weight_scheme = torch.per_channel_symmetric
        weight_observer = tq.PerChannelMinMaxObserver.with_args(
            dtype=weight_dtype,
            qscheme=weight_scheme,
            quant_min=weight_quant_min,
            quant_max=weight_quant_max,
        )
    elif weights_granularity == "per_tensor":
        weight_scheme = torch.per_tensor_symmetric
        weight_observer = tq.MinMaxObserver.with_args(
            dtype=weight_dtype,
            qscheme=weight_scheme,
            quant_min=weight_quant_min,
            quant_max=weight_quant_max,
        )
    else:
        raise ValueError(f"Weights granularity {weights_granularity} not supported - must be 'per_channel' or 'per_tensor'")

    # If Hardsigmoid module, use FixedParamsObserver since it has a fixed min/max mapping
    if module is not None and isinstance(module, nn.Hardsigmoid):
        return tq.QConfig(
            activation=tq.FixedQParamsObserver.with_args(
                dtype=torch.quint8,
                scale=1.0 / 256.0,
                zero_point=0,
            ),
            weight=weight_observer,
        )

    # Build QConfig for activations (non-Hardsigmoid layers)
    act_dtype = torch.qint8 if config["activations"]["dtype"] == "qint8" else torch.quint8
    act_scheme = torch.per_tensor_affine
    activations_observer = config['activations']['observer']
    if activations_observer == 'minmax':
        act_observer = tq.MinMaxObserver.with_args(
            dtype=act_dtype,
            qscheme=act_scheme,
            quant_min=act_quant_min,
            quant_max=act_quant_max,
        )
    elif activations_observer == 'histogram':
        act_observer = tq.HistogramObserver.with_args(
            dtype=act_dtype,
            qscheme=act_scheme,
            quant_min=act_quant_min,
            quant_max=act_quant_max,
        )
    else:
        raise ValueError(f"Activations observer {activations_observer} not supported - must be 'minmax' or 'histogram'")
    
    return tq.QConfig(activation=act_observer, weight=weight_observer)

def assign_bit_depth(model: torch.nn.Module, layer_entropies: dict, high_threshold: float, low_threshold: float) -> dict:
    """
    Assign bit depths to layers based on entropy thresholds. 8-bit for entropy >= high_threshold, 
    6-bit for low_threshold <= entropy < high_threshold, and 4-bit for entropy < low_threshold.
    
    Args:
        model: The model to validate layer names against
        layer_entropies: Dict mapping layer names to entropy values
        high_threshold: High entropy threshold
        low_threshold: Low entropy threshold
    
    Returns:
        Dict mapping layer names to bit depths (8, 6, or 4)
    """
    # get model layer names for validation
    model_layer_names = set(name for name, _ in model.named_modules())
    
    # assign bit depths based on entropy thresholds
    layer_bit_depths = {}
    unmatched_layers = []
    for layer_name, entropy in layer_entropies.items():
        if entropy >= high_threshold:
            layer_bit_depths[layer_name] = 8
        elif entropy >= low_threshold:
            layer_bit_depths[layer_name] = 6
        else:
            layer_bit_depths[layer_name] = 4
        
        # validate that layer exists in model
        if layer_name not in model_layer_names:
            unmatched_layers.append(layer_name)
    
    # warning for unmatched layers
    if unmatched_layers:
        print(f"Warning: {len(unmatched_layers)} entropy layer(s) could not be found in model modules")
    
    return layer_bit_depths

if __name__ == "__main__":
    set_seed()
    print("--- Running Mixed-Precision QAT ---")
    
    # Load config
    print("Loading Configuration ...")
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load entropies calculated in compute_layer_entropies.py
    entropy_file = "results/fp32_model_layer_entropies.json"
    print(f"Loading layer entropies from {entropy_file}...")
    with open(entropy_file, "r") as f:
        layer_entropies = json.load(f)
    
    # Calculate thresholds from percentiles (for now )
    entropy_values = list(layer_entropies.values())
    high_threshold = np.percentile(entropy_values, 75)
    low_threshold = np.percentile(entropy_values, 50)
    print(f"Entropy thresholds: high (75th percentile) = {high_threshold:.4f}, low (50th percentile) = {low_threshold:.4f}")
    
    # Load model
    qat_config = config["qat"]
    checkpoint_path = "models/finetuned_model_last_epoch.pth"
    print(f"Loading Model from checkpoint: {checkpoint_path} ...")
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    model = get_empty_model(num_classes=19)
    model = load_model(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()
    
    # Assign bit depths
    print(f"Assigning bit depths...")
    layer_bit_depths = assign_bit_depth(model, layer_entropies, high_threshold, low_threshold)
    
    # Count bit depths
    bit_counts = {8: 0, 6: 0, 4: 0}
    for bit_depth in layer_bit_depths.values():
        bit_counts[bit_depth] += 1
    print(f"Bit depth layer counts: 8-bit={bit_counts[8]}, 6-bit={bit_counts[6]}, 4-bit={bit_counts[4]}")
    print(f"Assigned bit depths to {len(layer_bit_depths)} layers")
    
    # Build QConfig for each layer
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
        qconfig = build_qconfig_per_layer(bit_depth, qat_config, module)
        qconfig_mapping.set_module_name(layer_name, qconfig)
    
    # Skip ASPP
    if qat_config.get("skip_aspp", False):
        qconfig_mapping.set_module_name("classifier.0", None) 
    
    # Load datasets
    cal_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_label_path = "data/gtFine_trainId/gtFine/train"
    val_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/val"
    val_label_path = "data/gtFine_trainId/gtFine/val"
    
    train_transforms = qat_config['training']['train_transforms']
    val_transforms = {'crop': False, 'resize': False, 'flip': False}
    train_dataset = cityScapesDataset(train_img_path, train_label_path, train_transforms)
    val_dataset = cityScapesDataset(val_img_path, val_label_path, val_transforms)
    cal_dataset = cityScapesDataset(train_img_path, train_label_path, train_transforms)

    batch_size = qat_config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    cal_loader = DataLoader(cal_dataset, batch_size=2, shuffle=True, drop_last=True)

    # Prepare model for mixed precision QAT
    example_inputs, _ = next(iter(cal_loader))
    example_inputs = example_inputs.to(device)
    prepared_model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, (example_inputs,))
    prepared_model = prepared_model.to(device)
    prepared_model.eval()
    
    # Setup training
    epochs = qat_config['training']['epochs']
    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(prepared_model.parameters(), 
                          lr=float(qat_config['training']['learning_rate']),
                          weight_decay=float(qat_config['training']['weight_decay']))
    scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-5)

    # Calibration
    if qat_config.get('calibration', {})['enabled']:
        print("Starting Calibration...")
        with torch.no_grad():
            for i, (image, _) in enumerate(cal_loader):
                image = image.to(device, non_blocking=True)
                prepared_model(image)
                if (i % 10 == 0 and i > 0):
                    print(f"  Calibrated {i} batches ...")
                if (i >= qat_config['calibration']['steps'] - 1):
                    print(f"  Completed {qat_config['calibration']['steps']} calibration steps.")
                    break
    
    # Train 
    print("Starting Mixed-Precision QAT...")
    best_val_loss = float('inf')
    training_history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        prepared_model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = prepared_model(images)['out']
            loss = loss_function(output, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        prepared_model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                output = prepared_model(images)['out']
                loss = loss_function(output, labels)
                val_loss += loss.item()

        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        # Store losses in dictionary
        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        
        # Always save latest model
        save_model(prepared_model, f"./models/mixed_precision_last_epoch.pth")
    
    # Verify quantization configs and fake quantization (QAT mode - weights are FP32 but fake quantized)
    print("\n" + "="*80)
    print("VERIFYING QUANTIZATION CONFIGS AND FAKE QUANTIZATION (QAT Mode)")
    print("="*80)
    print("NOTE: In QAT mode, weights are FP32 but fake quantization constrains values during forward pass")
    print("Checking that all layers have quantization configs and observers with correct bit depth ranges")
    print()
    
    # Define bit depth ranges for verification
    bit_depth_ranges = {
        8: (-128, 127),
        6: (-32, 31),
        4: (-8, 7)
    }
    
    # Check model structure for FakeQuantize modules and observers
    from torch.ao.quantization.fake_quantize import FakeQuantize
    from torch.ao.quantization.observer import ObserverBase
    
    layers_without_qconfig = []
    layers_with_qconfig = []
    layers_with_wrong_observer_range = []
    num_layers_to_show = 10
    layer_count = 0
    
    # Check all modules for quantization configs
    for name, module in prepared_model.named_modules():
        # Skip non-leaf modules
        if len(list(module.children())) > 0:
            continue
            
        # Check if module has a qconfig
        has_qconfig = hasattr(module, 'qconfig') and module.qconfig is not None
        
        if has_qconfig:
            layers_with_qconfig.append(name)
            
            # Try to find assigned bit depth
            assigned_bit_depth = None
            for layer_name, bit_depth in layer_bit_depths.items():
                if layer_name in name or name.endswith(layer_name.split('.')[-1]):
                    assigned_bit_depth = bit_depth
                    break
            
            if assigned_bit_depth is not None:
                # Check observer ranges
                weight_observer = module.qconfig.weight() if module.qconfig.weight else None
                act_observer = module.qconfig.activation() if module.qconfig.activation else None
                
                # Verify weight observer range
                if weight_observer is not None:
                    if hasattr(weight_observer, 'quant_min') and hasattr(weight_observer, 'quant_max'):
                        obs_min = weight_observer.quant_min
                        obs_max = weight_observer.quant_max
                        exp_min, exp_max = bit_depth_ranges[assigned_bit_depth]
                        
                        if obs_min != exp_min or obs_max != exp_max:
                            layers_with_wrong_observer_range.append((name, assigned_bit_depth, 'weight', obs_min, obs_max, exp_min, exp_max))
                
                # Show sample for first N layers
                if layer_count < num_layers_to_show:
                    print(f"Module: {name}")
                    print(f"  Type: {type(module).__name__}")
                    print(f"  Has QConfig: True")
                    if assigned_bit_depth:
                        print(f"  Assigned Bit Depth: {assigned_bit_depth}-bit")
                        if weight_observer:
                            print(f"  Weight Observer: {type(weight_observer).__name__}")
                            if hasattr(weight_observer, 'quant_min'):
                                print(f"    Range: [{weight_observer.quant_min}, {weight_observer.quant_max}]")
                        if act_observer:
                            print(f"  Activation Observer: {type(act_observer).__name__}")
                    print()
                    layer_count += 1
        else:
            # Check if this is a parameter layer that should have quantization
            has_params = any(p.requires_grad for p in module.parameters(recurse=False))
            if has_params:
                # Check if it's in layer_bit_depths (should be quantized)
                should_be_quantized = False
                for layer_name in layer_bit_depths.keys():
                    if layer_name in name or name.endswith(layer_name.split('.')[-1]):
                        should_be_quantized = True
                        break
                
                if should_be_quantized:
                    layers_without_qconfig.append(name)
    
    # Check for FakeQuantize modules (these are inserted by prepare_qat_fx)
    fake_quantize_modules = []
    for name, module in prepared_model.named_modules():
        if isinstance(module, FakeQuantize):
            fake_quantize_modules.append(name)
    
    # Report results
    print("="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    print(f"Total modules with QConfig: {len(layers_with_qconfig)}")
    print(f"Total FakeQuantize modules: {len(fake_quantize_modules)}")
    print()
    
    # Check for layers without QConfig (should be ZERO for layers that should be quantized)
    if layers_without_qconfig:
        print(f"ERROR: Found {len(layers_without_qconfig)} layers that should be quantized but have no QConfig:")
        for layer_name in layers_without_qconfig[:20]:
            print(f"  {layer_name}")
        if len(layers_without_qconfig) > 20:
            print(f"  ... and {len(layers_without_qconfig) - 20} more")
    else:
        print("✓ All layers that should be quantized have QConfig assigned")
    print()
    
    # Check for wrong observer ranges
    if layers_with_wrong_observer_range:
        print(f"WARNING: Found {len(layers_with_wrong_observer_range)} layers with observer ranges that don't match assigned bit depth:")
        for name, bit_depth, obs_type, obs_min, obs_max, exp_min, exp_max in layers_with_wrong_observer_range[:10]:
            print(f"  {name} ({bit_depth}-bit {obs_type}): observer range [{obs_min}, {obs_max}], expected [{exp_min}, {exp_max}]")
        if len(layers_with_wrong_observer_range) > 10:
            print(f"  ... and {len(layers_with_wrong_observer_range) - 10} more")
    else:
        print("✓ All observers have correct bit depth ranges")
    print()
    
    # Check max bit depth
    max_bit_depth = max(layer_bit_depths.values()) if layer_bit_depths else 8
    if max_bit_depth > 8:
        print(f"WARNING: Maximum bit depth is {max_bit_depth}, but should be ≤ 8")
    else:
        print(f"✓ Maximum bit depth is {max_bit_depth}-bit (≤ 8-bit)")
    print()
    
    # Verify weights are FP32 (expected in QAT mode)
    print("Verifying weight storage (should be FP32 in QAT mode):")
    fp32_weight_layers = []
    quantized_weight_layers = []
    for name, param in prepared_model.named_parameters():
        if param.requires_grad:
            is_quantized = hasattr(param, 'int_repr')
            if is_quantized:
                quantized_weight_layers.append(name)
            else:
                fp32_weight_layers.append(name)
    
    print(f"  FP32 weight layers: {len(fp32_weight_layers)} (expected in QAT mode)")
    print(f"  Quantized weight layers: {len(quantized_weight_layers)} (unexpected in QAT mode)")
    if quantized_weight_layers:
        print(f"  WARNING: Found quantized weights in QAT mode (should be FP32):")
        for layer_name in quantized_weight_layers[:10]:
            print(f"    {layer_name}")
    
    print("="*80)
    print()

    # calculate model size 
    print(f"Calculating model size...")
    model_size_mb = calculate_model_size_mixed_precision(prepared_model, layer_bit_depths=layer_bit_depths)
    print(f"\nModel Size: {model_size_mb:.2f} MB")
    
    # Run inference on full validation set and calculate mIoU
    print(f"\nRunning inference on validation set...")
    prepared_model = prepared_model.to(device)
    prepared_model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, (image, labels) in enumerate(tqdm(val_loader, desc="Validation inference")):
            image = image.to(device, non_blocking=True)
            
            out = prepared_model(image)['out']
            preds = out.argmax(dim=1)
            
            all_predictions.append(preds.cpu())
            all_targets.append(labels)
            
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate mIoU
    print(f"\nCalculating mIoU...")
    miou, per_class_ious = calculate_miou(all_predictions, all_targets, num_classes=19, ignore_index=255)
    print(f"mIoU: {miou:.4f}")

    # Save prepared model (QAT mode)
    output_path = "models/mixed_precision_model.pth"
    save_model(prepared_model, output_path)
    print(f"Mixed-precision model (QAT mode) saved to {output_path}")
    
    # Save training history
    history_output = "results/mixed_precision_training_history.json"
    with open(history_output, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to {history_output}")
    
    # Save bit depth assignment
    bit_depth_output = "results/mixed_precision_layer_bit_depths.json"
    with open(bit_depth_output, "w") as f:
        json.dump(layer_bit_depths, f, indent=2)
    print(f"Bit depth assignments saved to {bit_depth_output}")
    
    # Save evaluation results to text file
    results_output = "results/mixed_precision_results.txt"
    with open(results_output, "a") as f:
        f.write(f"Model Size: {model_size_mb:.2f} MB\n\n")
        
        f.write(f"mIoU: {miou:.4f}\n\n")
        
        f.write("Per-Class IoU:\n")
        for class_idx, iou in enumerate(per_class_ious):
            f.write(f"  Class {class_idx}: {iou:.4f}\n")
        f.write("\n")
        
        f.write("Training Losses:\n")
        for epoch, loss in enumerate(training_history["train_loss"], 1):
            f.write(f"  Epoch {epoch}: {loss:.6f}\n")
        f.write("\n")
        
        f.write("Validation Losses:\n")
        for epoch, loss in enumerate(training_history["val_loss"], 1):
            f.write(f"  Epoch {epoch}: {loss:.6f}\n")
        f.write("\n")
        
        f.write("Bit Depth Distribution:\n")
        f.write(f"  8-bit layers: {bit_counts[8]}\n")
        f.write(f"  6-bit layers: {bit_counts[6]}\n")
        f.write(f"  4-bit layers: {bit_counts[4]}\n")
        f.write("\n")
        
        f.write("Configuration Parameters:\n")
        f.write(yaml.dump(qat_config, default_flow_style=False, sort_keys=False))
        f.write("\n" + "="*60 + "\n\n")

    # Visualization
    # Load Image
    sample_folder = "frankfurt"
    sample_id = "000001_014565"
    image_path = f"./data/leftImg8bit_trainvaltest/leftImg8bit/val/{sample_folder}/{sample_folder}_{sample_id}_leftImg8bit.png"
    ground_truth_path = f"./data/gtFine_trainIdColorized/gtFine/val/{sample_folder}/{sample_folder}_{sample_id}_gtFine_color.png"
    image = Image.open(image_path).convert("RGB")
    
    # Normalize image
    normalize_image = T.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    image_tensor = F.to_tensor(image)
    image_tensor = normalize_image(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Load FP32 model for comparison
    print("Loading FP32 model for comparison...")
    fp32_model = get_empty_model(num_classes=19)
    fp32_checkpoint = "models/finetuned_model_last_epoch.pth"
    fp32_model = load_model(fp32_model, fp32_checkpoint, device=device)
    fp32_model = fp32_model.to(device)
    fp32_model.eval()

    # Run inference on both models
    with torch.no_grad():
        # FP32 model prediction
        fp32_output = fp32_model(image_tensor)['out']
        fp32_predicted_mask = torch.argmax(fp32_output.squeeze(), dim=0).detach().cpu().numpy()
        
        # Mixed-precision model prediction
        prepared_model.eval()  # Set to eval for inference
        prepared_model = prepared_model.to(device)
        mp_output = prepared_model(image_tensor)['out']
        mp_predicted_mask = torch.argmax(mp_output.squeeze(), dim=0).detach().cpu().numpy()

    # Map train IDs to colors for visualization
    def mask_to_color(mask):
        height, width = mask.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for train_id, color in map_train_id_to_color.items():
            color_mask[mask == train_id] = color
        return color_mask

    fp32_color_mask = mask_to_color(fp32_predicted_mask)
    mp_color_mask = mask_to_color(mp_predicted_mask)

    # Display comparison
    plt.figure(figsize=(20, 4), constrained_layout=True)
    
    plt.subplot(1, 5, 1)
    plt.title("Original Image", fontsize=12)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.title("Ground Truth", fontsize=12)
    ground_truth = Image.open(ground_truth_path)
    plt.imshow(ground_truth)
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.title("FP32 Model Prediction", fontsize=12)
    plt.imshow(fp32_color_mask)
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.title("Mixed-Precision Model Prediction", fontsize=12)
    plt.imshow(mp_color_mask)
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.title("Difference (FP32 vs MP)", fontsize=12)
    # Show pixels where predictions differ
    difference_mask = (fp32_predicted_mask != mp_predicted_mask).astype(np.uint8) * 255
    plt.imshow(difference_mask, cmap='hot')
    plt.axis("off")
    
    plt.savefig("./results/mp_vs_fp32_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
