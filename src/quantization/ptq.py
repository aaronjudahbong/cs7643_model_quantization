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

DEBUG = True

SEED = 42

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

def set_seed(seed = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def build_qconfig(config, mode):
    qconfig = tq.get_default_qconfig_mapping("fbgemm")
    if mode == "default":
        return qconfig
    
    # Define weight and activation dtype.
    weight_dtype = torch.qint8 if config["weights"]["dtype"] == "qint8" else torch.quint8
    act_dtype = torch.qint8 if config["activations"]["dtype"] == "qint8" else torch.quint8
    
    # Define weight scheme based on granularity (keep it symmetric).
    # Keep activation scheme fixed (affine).
    weight_scheme = torch.per_channel_symmetric if config["weights"]["granularity"] == "per_channel" else torch.per_tensor_symmetric
    act_scheme = torch.per_tensor_affine

    # Define quantization ranges based on model.
    if mode == "int8":
        weight_quant_min, weight_quant_max = -128, 127
        act_quant_min, act_quant_max = 0, 127 # reduce_range = True
    elif mode == "int6":
        weight_quant_min, weight_quant_max = -32, 31
        act_quant_min, act_quant_max = 0, 31 # reduce_range = True
    elif mode == "int4":
        weight_quant_min, weight_quant_max = -8, 7
        act_quant_min, act_quant_max = 0, 7 # reduce_range = True

    # Build QConfig for weights and activations.
    # If per_channel need to use PerChannelMinMaxObserver, else use MinMaxObserver.
    if config["weights"]["granularity"] == "per_channel":
        weight_observer = tq.PerChannelMinMaxObserver.with_args(
            dtype=weight_dtype,
            qscheme=weight_scheme,
            quant_min=weight_quant_min,
            quant_max=weight_quant_max,
        )
    else:
        weight_observer = tq.MinMaxObserver.with_args(
            dtype=weight_dtype,
            qscheme=weight_scheme,
            quant_min=weight_quant_min,
            quant_max=weight_quant_max,
        )

    act_observer = tq.HistogramObserver.with_args(
        dtype=act_dtype,
        qscheme=act_scheme,
        quant_min=act_quant_min,
        quant_max=act_quant_max,
    )

    global_config = tq.QConfig(activation=act_observer, weight=weight_observer)
    qconfig_mapping = tq.QConfigMapping().set_global(global_config)

    fixed_qconfig = tq.QConfig(
        activation = tq.FixedQParamsObserver.with_args(
            dtype= torch.quint8,
            scale = 1.0 / 256.0,
            zero_point = 0,
        ),
        weight = weight_observer,
    )
    
    # Hardsigmoid has a fixed min/max mapping (need to use FixedQParamsObserver).
    qconfig_mapping.set_object_type(nn.Hardsigmoid, fixed_qconfig)

    if config["skip_aspp"]: 
        # Skip quantizing the ASPP module (only quantize the backbone).
        qconfig_mapping.set_module_name("classifier.0", None) # ASPP module (convs and project).

    return qconfig_mapping

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
    qconfig_mapping = build_qconfig(config, config["mode"])

    print("Quantizing Model ...")
    # Grab a sample input from validation set with CalibrationDataset.
    dataroot = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    calib_dataset = CalibrationDataset(dataroot)
    calib_dataloader = DataLoader(calib_dataset, batch_size=1, shuffle=True)

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

    print("Saving Quantized Model ...")
    torch.save(quantized_model.state_dict(), "models/ptq_quantized_model.pth")
    print(f"Quantized Model Size (MB): {os.path.getsize('models/ptq_quantized_model.pth') / 1e6:.2f}")

    print("Saving model as TorchScript ...")
    traced_model = torch.jit.trace(QuantizedModelWrapper(quantized_model), (sample_tensor,))
    torch.jit.save(traced_model, "models/ptq_quantized_model_scripted.pt")
    print(f"Scripted Quantized Model Size (MB): {os.path.getsize('models/ptq_quantized_model_scripted.pt') / 1e6:.2f}")