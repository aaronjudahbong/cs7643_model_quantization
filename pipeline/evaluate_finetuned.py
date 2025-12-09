import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.deeplabv3_mnv3 import get_empty_model, load_model
from pipeline.create_dataset import cityScapesDataset
from pipeline.metrics import calculate_miou

if __name__ == "__main__":
    print("--- Evaluating Finetuned Model ---")
    
    # Device setup
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Load model
    model_checkpoint = "models/finetuned_model_last_epoch.pth"
    print(f"Loading model from: {model_checkpoint}")
    model = get_empty_model(num_classes=19)
    model = load_model(model, model_checkpoint, device=device)
    model = model.to(device)
    model.eval()
    
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
            
            output = model(images)['out']
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

