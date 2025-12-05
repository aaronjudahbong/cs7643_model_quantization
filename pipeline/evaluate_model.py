import torch
from torch.utils.data import DataLoader
from pipeline.create_dataset import cityScapesDataset
from pipeline.metrics import calculate_miou, calculate_model_size, measure_inference_latency
from src.models.deeplabv3_mnv3 import get_empty_model, load_model
import argparse
import os

# Set default paths 
validation_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/val"
validation_label_folder = "./data/gtFine_trainId/gtFine/val"

def evaluate_model(model_path: str, val_image_folder: str, val_label_folder: str, 
                   device: str = None, batch_size: int = 1, quantization_bits: int = 8) -> dict:
    """
    Evaluate a model checkpoint with metrics mIoU, model size, and inference latency.
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        val_image_folder: Path to validation images
        val_label_folder: Path to validation labels
        device: Device to use ('cpu', 'cuda', 'mps', or None)
        batch_size: Batch size for inference
        quantization_bits: Quantization bits for model size calculation
    
    Returns: 
        Dictionary with metrics:
            'miou': mIoU score
            'model_size_mb': model size in MB
            'latency_stats': Dictionary with latency statistics (mean, std, median) in milliseconds
    """
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = get_empty_model(num_classes=19) # get model architecture
    model = load_model(model, model_path, device=device) # load model weights 
    model = model.to(device)
    model.eval()
    
    # Calculate model size
    model_size_mb = calculate_model_size(model, quantization_bits=quantization_bits)
    print(f"\nModel Size: {model_size_mb:.2f} MB")
    
    # Load validation dataset
    print(f"\nLoading validation dataset...")
    val_transforms = {'crop': False, 'resize': False, 'flip': False}
    val_dataset = cityScapesDataset(val_image_folder, val_label_folder, val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Measure inference latency on a dummy sample
    print(f"\nMeasuring inference latency...")
    sample_image, _ = next(iter(val_loader))
    sample_image = sample_image.to(device, non_blocking=True)
    latency_stats = measure_inference_latency(
        model, 
        sample_image.shape[1:],  # (C, H, W)
        device,
        num_warmup=10,
        num_runs=100
    )
    print(f"Inference Latency:")
    print(f"  Mean: {latency_stats['mean_ms']:.2f} ± {latency_stats['std_ms']:.2f} ms")
    print(f"  Median: {latency_stats['median_ms']:.2f} ms")
    
    # Run inference on full validation set and calculate mIoU
    print(f"\nRunning inference on validation set...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, (image, labels) in enumerate(val_loader):
            image = image.to(device, non_blocking=True)
            
            out = model(image)['out']
            preds = out.argmax(dim=1)
            
            all_predictions.append(preds.cpu())
            all_targets.append(labels)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(val_loader)}")
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate mIoU
    print(f"\nCalculating mIoU...")
    miou, iou = calculate_miou(all_predictions, all_targets, num_classes=19, ignore_index=255)
    print(f"mIoU: {miou:.4f}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Inference Latency: {latency_stats['mean_ms']:.2f} ± {latency_stats['std_ms']:.2f} ms")
    print(f"mIoU: {miou:.4f}")
    print(f"{'='*50}")
    
    return {
        'miou': miou,
        'iou': iou,
        'model_size_mb': model_size_mb,
        'latency_stats': latency_stats
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                    help='Path to model checkpoint (.pth file)')
    parser.add_argument('--val_images', type=str, 
                    default=validation_image_folder,
                    help='Path to validation images folder')
    parser.add_argument('--val_labels', type=str,
                    default=validation_label_folder,
                    help='Path to validation labels folder')
    parser.add_argument('--device', type=str, default=None,
                    choices=['cpu', 'cuda', 'mps', None],
                    help='Device to use (auto-detect if not specified)')
    parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for inference')
    parser.add_argument('--quantization_bits', type=int, default=8,
                    help='Quantization bits for model size calculation')
    args = parser.parse_args()
    
    evaluation = evaluate_model(
        model_path=args.model,
        val_image_folder=args.val_images,
        val_label_folder=args.val_labels,
        device=args.device,
        batch_size=args.batch_size,
        quantization_bits=args.quantization_bits
    )

    miou = evaluation['miou']
    iou = evaluation['iou']
    model_size = evaluation['model_size_mb']
    latency = evaluation['latency_stats']
    model_name = (os.path.basename(args.model)).split(".")[0]

    with open(f"./results/{model_name}_evaluation.txt", "a") as f:
        f.write(
            f"mIoU: {miou}, \n"
            f"Class 0 IoU: {iou[0]}, \n"
            f"Class 1 IoU: {iou[1]}, \n"
            f"Class 2 IoU: {iou[2]}, \n"
            f"Class 3 IoU: {iou[3]}, \n"
            f"Class 4 IoU: {iou[4]}, \n"
            f"Class 5 IoU: {iou[5]}, \n"
            f"Class 6 IoU: {iou[6]}, \n"
            f"Class 7 IoU: {iou[7]}, \n"
            f"Class 8 IoU: {iou[8]}, \n"
            f"Class 9 IoU: {iou[9]}, \n"
            f"Class 10 IoU: {iou[10]}, \n"
            f"Class 11 IoU: {iou[11]}, \n"
            f"Class 12 IoU: {iou[12]}, \n"
            f"Class 13 IoU: {iou[13]}, \n"
            f"Class 14 IoU: {iou[14]}, \n"
            f"Class 15 IoU: {iou[15]}, \n"
            f"Class 16 IoU: {iou[16]}, \n"
            f"Class 17 IoU: {iou[17]}, \n"
            f"Class 18 IoU: {iou[18]}, \n\n"
            f"Model Size (MB): {model_size}, \n"
            f"Latency: {latency}\n"
        )
