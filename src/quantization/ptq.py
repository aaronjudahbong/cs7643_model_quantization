import os
import yaml
import torch
import torchvision
import torch.nn as nn
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
from torchvision.transforms import functional as F
from PIL import Image

from src.models.deeplabv3_mnv3 import get_empty_model, load_model

DEBUG = True

if __name__ == "__main__":
    print("Running PTQ Script")
    print("Load PTQ Configuration File")
    # Load configuration.
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)["ptq"]

    print("Load Model with defined Checkpoint")
    # Get empty model and load checkpoint weights.
    model = get_empty_model()
    model = load_model(model, config["model_checkpoint"])
    model = model.eval()

    if DEBUG:
        print("Baseline FP32 Model: ")
        print(model)

    # Get Default Quantization Configuration.
    qconfig = tq.get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig} 

    print("Prepare Model for PTQ")
    # Grab a sample input from validation set.
    sample_img = Image.open("data/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png").convert('RGB')
    sample_tensor = F.to_tensor(sample_img).unsqueeze(0)
    prepared_model = quantize_fx.prepare_fx(model, qconfig_dict, sample_tensor)

    print("Quantizing Model")
    quantized_model = quantize_fx.convert_fx(prepared_model)

    if DEBUG:
        print("Quantized Model: ")
        print(quantized_model)


