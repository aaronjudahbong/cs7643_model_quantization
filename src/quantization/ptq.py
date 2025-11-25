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

    # Get Default Quantization Configuration (for now).
    qconfig_mapping = tq.get_default_qconfig_mapping("fbgemm")

    print("Quantizing Model ...")
    # Grab a sample input from validation set.
    sample_img = Image.open("data/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png").convert('RGB')
    sample_tensor = F.to_tensor(sample_img).unsqueeze(0)
    prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, (sample_tensor,))
    quantized_model = quantize_fx.convert_fx(prepared_model)

    print("Saving Quantized Model ...")
    torch.save(quantized_model.state_dict(), "models/ptq_quantized_model.pth")
    print(f"Quantized Model Size (MB): {os.path.getsize('models/ptq_quantized_model.pth') / 1e6:.2f}")

    print("Saving model as TorchScript ...")
    traced_model = torch.jit.trace(QuantizedModelWrapper(quantized_model), (sample_tensor,))
    torch.jit.save(traced_model, "models/ptq_quantized_model_scripted.pt")
    print(f"Scripted Quantized Model Size (MB): {os.path.getsize('models/ptq_quantized_model_scripted.pt') / 1e6:.2f}")