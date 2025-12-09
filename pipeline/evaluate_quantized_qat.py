import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.ao.quantization.quantize_fx as quantize_fx
from src.models.deeplabv3_mnv3 import get_empty_model, load_model
from src.quantization.quantization_utils import build_qconfig
from pipeline.create_dataset import cityScapesDataset
from pipeline.metrics import calculate_miou

if __name__ == "__main__":
    print("--- Evaluating Quantized QAT Model ---")
    
    # Device setup
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Load config to get QConfig
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)["qat"]
    
    # Load base model (needed to create quantized structure)
    base_model_path = config['model_checkpoint']
    print(f"Loading base model from: {base_model_path}")
    model = get_empty_model(num_classes=19)
    model = load_model(model, base_model_path, device=device)
    model = model.to(device)
    model.eval()
    
    # Build QConfig (same as used during QAT)
    print(f"Building QConfig with mode: {config['mode']}...")
    qconfig_mapping = build_qconfig("qat", config)
    
    # Prepare sample input for quantization
    cal_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_label_path = "data/gtFine_trainId/gtFine/train"
    cal_dataset = cityScapesDataset(cal_img_path, train_label_path, config['training']['train_transforms'])
    cal_loader = DataLoader(cal_dataset, batch_size=2, shuffle=True, drop_last=True)
    
    example_image, _ = next(iter(cal_loader))
    example_image = example_image.to(device)
    
    # Prepare and convert model to get quantized structure
    print("Preparing model for quantization...")
    prepared_model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, (example_image,))
    prepared_model = prepared_model.cpu()
    quantized_model = quantize_fx.convert_fx(prepared_model.eval())
    
    # Load quantized state_dict
    quantized_model_path = "qat_results/qat_quantized_model_0.pth"
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
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    
    # Run inference on validation set
    print(f"\nRunning inference on validation set...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation inference"):
            images = images.to(device, non_blocking=True)
            
            output = quantized_model(images)['out']
            preds = output.argmax(dim=1)
            
            all_predictions.append(preds.cpu())
            all_targets.append(labels)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate mIoU
    print(f"\nCalculating mIoU...")
    miou, per_class_ious = calculate_miou(all_predictions, all_targets, num_classes=19, ignore_index=255)
    print(f"Final mIoU: {miou:.4f}")
    
    # Print per-class IoUs
    print("\nPer-Class IoU:")
    for class_idx, iou in enumerate(per_class_ious):
        print(f"  Class {class_idx}: {iou:.4f}")

