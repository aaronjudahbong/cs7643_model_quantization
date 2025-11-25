import os
import yaml
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
from torchvision.transforms import functional as F
from PIL import Image

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from src.models.deeplabv3_mnv3 import get_empty_model, load_model

DEBUG = True

class QuantizedModelWrapper(nn.Module):
    # Wrapper to output `Tensor` instead of a `dict`.
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)["out"]
    
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
    return {"": global_config}

if __name__ == "__main__":
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
    # Grab a sample input from validation set (NEED TO MODIFY THE SIZE maybe).
    sample_img = Image.open("data/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png").convert('RGB')
    sample_tensor = F.to_tensor(sample_img).unsqueeze(0)
    prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, (sample_tensor,))
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