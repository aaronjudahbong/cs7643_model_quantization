import os
import glob
import yaml
from PIL import Image

# Torch imports.
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from src.models.deeplabv3_mnv3 import get_empty_model, load_model
from .quantization_utils import set_seed, build_qconfig
from pipeline.evaluate_ptq_model import evaluate_model

DEBUG = True

# Set default paths 
validation_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/val"
validation_label_folder = "./data/gtFine_trainId/gtFine/val"

class CalibrationDataset(Dataset):
    # Similar to cityScapesDataset (create_dataset.py) but only returns images.
    # Need to make sure Transform is the SAME.
    def __init__(self, img_dir):
        self.img_paths = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image

class QuantizedModelWrapper(nn.Module):
    # Wrapper to output `Tensor` instead of a `dict`.
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)["out"]

if __name__ == "__main__":
    set_seed()
    print("--- Running PTQ Script ---")
    print("Loading Configuration ...")
    # Load configuration.
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)["ptq"]

    print(f"Loading Model from checkpoint: {config['model_checkpoint']} ...")
    # Get empty model and load checkpoint weights.
    model = get_empty_model()
    model = load_model(model, config["model_checkpoint"])
    model.eval()
    print(f"Baseline Model Size (MB): {os.path.getsize(config['model_checkpoint']) / 1e6:.2f}")

    with open("ptq_base.txt", "w") as f:
        print(model, file=f)

    # Get Default Quantization Configuration (for now).
    print(f"Building QConfig with mode: {config['mode']}...")
    qconfig_mapping = build_qconfig("ptq", config)

    print("Quantizing Model ...")
    # Grab a sample input from validation set with CalibrationDataset.
    dataroot = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    calib_dataset = CalibrationDataset(dataroot)
    calib_dataloader = DataLoader(calib_dataset, batch_size=1, shuffle=False)

    sample_tensor = next(iter(calib_dataloader))
    prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, (sample_tensor,))
    prepared_model.eval()

    # Calibrate with Calibration Dataset.
    print("Calibrating Model ...")
    with torch.no_grad():
        for i, image in enumerate(calib_dataloader):
            prepared_model(image)
            if (i % 10 == 0 and i > 0):
                print(f"  Calibrated {i} batches ...")

            if (i >= config["calibration"]["steps"] - 1):
                print(f"  Completed {config['calibration']['steps']} calibration steps.")
                break
    quantized_model = quantize_fx.convert_fx(prepared_model)
    
    with open("ptq_quantized.txt", "w") as f:
        print(quantized_model, file=f)
        
    # Metrics
    evaluation = evaluate_model(
        model = QuantizedModelWrapper(quantized_model).to("cpu").eval(),
        val_image_folder = validation_image_folder,
        val_label_folder = validation_label_folder,
        device = "cpu",
        batch_size = 1,
        quantization_bits = 8
    )
    
    with open("ptq_evaluation_metrics.txt", "w") as f:
        f.write(
            f"mIoU: {evaluation['miou']}, \n"
            f"Class 0 IoU: {evaluation['iou'][0]}, \n"
            f"Class 1 IoU: {evaluation['iou'][1]}, \n"
            f"Class 2 IoU: {evaluation['iou'][2]}, \n"
            f"Class 3 IoU: {evaluation['iou'][3]}, \n"
            f"Class 4 IoU: {evaluation['iou'][4]}, \n"
            f"Class 5 IoU: {evaluation['iou'][5]}, \n"
            f"Class 6 IoU: {evaluation['iou'][6]}, \n"
            f"Class 7 IoU: {evaluation['iou'][7]}, \n"
            f"Class 8 IoU: {evaluation['iou'][8]}, \n"
            f"Class 9 IoU: {evaluation['iou'][9]}, \n"
            f"Class 10 IoU: {evaluation['iou'][10]}, \n"
            f"Class 11 IoU: {evaluation['iou'][11]}, \n"
            f"Class 12 IoU: {evaluation['iou'][12]}, \n"
            f"Class 13 IoU: {evaluation['iou'][13]}, \n"
            f"Class 14 IoU: {evaluation['iou'][14]}, \n"
            f"Class 15 IoU: {evaluation['iou'][15]}, \n"
            f"Class 16 IoU: {evaluation['iou'][16]}, \n"
            f"Class 17 IoU: {evaluation['iou'][17]}, \n"
            f"Class 18 IoU: {evaluation['iou'][18]}, \n"
            f"Model Size (MB): {evaluation['model_size_mb']}, \n"
        )

    print("Saving Quantized Model ...")
    torch.save(quantized_model.state_dict(), "models/ptq_quantized_model.pth")
    print(f"Quantized Model Size (MB): {os.path.getsize('models/ptq_quantized_model.pth') / 1e6:.2f}")

    print("Saving model as TorchScript ...")
    traced_model = torch.jit.trace(QuantizedModelWrapper(quantized_model), (sample_tensor,))
    torch.jit.save(traced_model, "models/ptq_quantized_model_scripted.pt")
    print(f"Scripted Quantized Model Size (MB): {os.path.getsize('models/ptq_quantized_model_scripted.pt') / 1e6:.2f}")