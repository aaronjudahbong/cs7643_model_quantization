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

DEBUG = True

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

    # Load configuration.
    print("Loading Configuration ...")
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)["ptq"]
    
    print(f"Quantizing to {config['mode']}")

    # Calibration settings to sweep
    calib_steps_list = config["calibration"]["steps"]
    
    # Calibration dataset
    dataroot = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    calib_dataset = CalibrationDataset(dataroot)
    calib_dataloader = DataLoader(calib_dataset, batch_size=1, shuffle=True)

    print("\nStarting PTQ sweep...\n")

    for steps in calib_steps_list:
        print(f" Running PTQ for {steps} calibration steps")

        # Load a fresh FP32 model each run
        model = get_empty_model()
        model = load_model(model, config["model_checkpoint"])
        model.eval()

        # Build QConfig
        qconfig_mapping = build_qconfig("ptq", config)

        # Prepare (FX graph rewrite)
        sample_tensor = next(iter(calib_dataloader))
        prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, (sample_tensor,))
        prepared_model.eval()

        # Calibrate
        print("Calibrating...")
        with torch.no_grad():
            for i, image in enumerate(calib_dataloader):
                prepared_model(image)

                if i % 10 == 0 and i > 0:
                    print(f"  Collected stats from {i} batches...")

                if i >= steps - 1:
                    print(f"  Completed {steps} calibration steps.")
                    break

        # Convert to fully quantized INT8 model
        print("Converting to quantized model...")
        quantized_model = quantize_fx.convert_fx(prepared_model)

        # File naming
        suffix = f"_steps_{steps}"
        model_path = f"models/ptq_quantized{suffix}.pth"
        script_path = f"models/ptq_quantized_scripted{suffix}.pt"
        txt_path = f"ptq_quantized{suffix}.txt"

        # Save readable FX graph
        with open(txt_path, "w") as f:
            print(quantized_model, file=f)

        # Save state_dict
        torch.save(quantized_model.state_dict(), model_path)
        print(f"Quantized Model Saved → {model_path}")

        # Save TorchScript version
        traced_model = torch.jit.trace(QuantizedModelWrapper(quantized_model), (sample_tensor,))
        torch.jit.save(traced_model, script_path)
        print(f"TorchScript Model Saved → {script_path}")

        # Size reporting
        print(f"Model size (MB): {os.path.getsize(model_path) / 1e6:.2f}")
        print(f"Scripted size (MB): {os.path.getsize(script_path) / 1e6:.2f}")