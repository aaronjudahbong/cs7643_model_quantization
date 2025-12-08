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

def evaluate_model(model, val_image_folder: str, val_label_folder: str, 
                   device: str = None, batch_size: int = 1, quantization_bits: int = 8) -> dict:
    """
    Evaluate a model checkpoint with streaming mIoU calculation (memory-safe).

    Returns: 
        {
            'miou': float,
            'iou': list of per-class IoUs,
            'model_size_mb': float,
            'latency_stats': placeholder
        }
    """

    import numpy as np

    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Model size
    # --------------------------------------------------
    model_size_mb = calculate_model_size(model, quantization_bits=quantization_bits)
    print(f"\nModel Size: {model_size_mb:.2f} MB")

    # --------------------------------------------------
    # Dataset loader
    # --------------------------------------------------
    print(f"\nLoading validation dataset...")
    val_transforms = {'crop': False, 'resize': False, 'flip': False}
    val_dataset = cityScapesDataset(val_image_folder, val_label_folder, val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --------------------------------------------------
    # Prepare streaming confusion matrix
    # --------------------------------------------------
    num_classes = 19
    ignore_index = 255
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    # --------------------------------------------------
    # Inference loop (streaming, no prediction storage)
    # --------------------------------------------------
    print(f"\nRunning inference on validation set...")

    model.eval()
    with torch.no_grad():
        for i, (image, labels) in enumerate(val_loader):

            image = image.to(device, non_blocking=True)

            # forward pass
            out = model(image)
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

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"mIoU: {miou:.4f}")
    print(f"{'='*50}")

    return {
        'miou': float(miou),
        'iou': iou.tolist(),
        'model_size_mb': model_size_mb,
        'latency_stats': {}
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

    # miou = evaluation['miou']
    # iou = evaluation['iou']
    # model_size = evaluation['model_size_mb']
    # latency = evaluation['latency_stats']
    # model_name = (os.path.basename(args.model)).split(".")[0]

    # with open(f"./results/{model_name}_evaluation.txt", "a") as f:
    #     f.write(
    #         f"mIoU: {miou}, \n"
    #         f"Class 0 IoU: {iou[0]}, \n"
    #         f"Class 1 IoU: {iou[1]}, \n"
    #         f"Class 2 IoU: {iou[2]}, \n"
    #         f"Class 3 IoU: {iou[3]}, \n"
    #         f"Class 4 IoU: {iou[4]}, \n"
    #         f"Class 5 IoU: {iou[5]}, \n"
    #         f"Class 6 IoU: {iou[6]}, \n"
    #         f"Class 7 IoU: {iou[7]}, \n"
    #         f"Class 8 IoU: {iou[8]}, \n"
    #         f"Class 9 IoU: {iou[9]}, \n"
    #         f"Class 10 IoU: {iou[10]}, \n"
    #         f"Class 11 IoU: {iou[11]}, \n"
    #         f"Class 12 IoU: {iou[12]}, \n"
    #         f"Class 13 IoU: {iou[13]}, \n"
    #         f"Class 14 IoU: {iou[14]}, \n"
    #         f"Class 15 IoU: {iou[15]}, \n"
    #         f"Class 16 IoU: {iou[16]}, \n"
    #         f"Class 17 IoU: {iou[17]}, \n"
    #         f"Class 18 IoU: {iou[18]}, \n\n"
    #         f"Model Size (MB): {model_size}, \n"
    #         f"Latency: {latency}\n"
    #     )
