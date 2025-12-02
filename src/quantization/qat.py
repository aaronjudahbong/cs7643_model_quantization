import os
import glob
import yaml
from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F

from src.models.deeplabv3_mnv3 import get_empty_model, load_model
from .quantization_utils import set_seed, build_qconfig
from pipeline.create_dataset import cityScapesDataset

class TrainValDataset(Dataset):
    # Similar to cityScapesDataset (create_dataset.py) but only returns images.
    # Need to make sure Transform is the SAME.
    def __init__(self, img_dir, label_dir):
        self.img_paths = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
        self.img_paths = sorted(self.img_paths)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.label_paths = glob.glob(os.path.join(label_dir, "**", "*.png"), recursive=True)
        self.label_paths = sorted(self.label_paths)
        assert len(self.img_paths) == len(self.label_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transform(image)

        label = torch.from_numpy(np.array(Image.open(self.label_paths[idx]))).long()

        return image, label

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

if __name__ == "__main__":
    set_seed()
    print("--- Running QAT Script ---")
    print("Loading Configuration ...")
    # Load configuration.
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)["qat"]

    print(f"Loading Model from checkpoint: {config['model_checkpoint']} ...")
    # Get empty model and load checkpoint weights.
    model = get_empty_model()
    model = load_model(model, config["model_checkpoint"])
    model.eval()
    print(f"Baseline Model Size (MB): {os.path.getsize(config['model_checkpoint']) / 1e6:.2f}")

    # Get Quantization Configuration
    print(f"Building QConfig with mode: {config['mode']}...")
    qconfig_mapping = build_qconfig("qat", config)

    cal_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_label_path = "data/gtFine_trainId/gtFine/train"
    val_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/val"
    val_label_path = "data/gtFine_trainId/gtFine/val"

    cal_dataset = CalibrationDataset(cal_img_path)
    train_dataset = TrainValDataset(train_img_path, train_label_path)
    val_dataset = TrainValDataset(val_img_path, val_label_path)

    cal_dataloader = DataLoader(cal_dataset, batch_size=2, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    example_inputs = next(iter(cal_dataloader))
    prepared_model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, (example_inputs,))
    prepared_model.train()

    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(prepared_model.parameters(), lr=float(config['training']['learning_rate']),
                                                        weight_decay=float(config['training']['weight_decay']))

    print("Start Calibration...")
    if config['calibration']['enabled']:
        with torch.no_grad():
            for i, image in enumerate(cal_dataloader):
                prepared_model(image)
                if (i % 10 == 0 and i > 0):
                    print(f"  Calibrated {i} batches ...")

                if (i >= config['calibration']['steps'] - 1):
                    print(f"  Completed {config['calibration']['steps']} calibration steps.")
                    break


    print("Starting QAT...")
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        prepared_model.train()
        training_loss = 0
        for i, (image, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            out = prepared_model(image)['out']
            loss = loss_function(out, label)
            training_loss += loss
            loss.backward()
            optimizer.step()

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for i, (image, label) in enumerate(val_dataloader):
                out = model(image)['out']
                loss = loss_function(out, label)
                validation_loss += loss

        if (i % 10 == 0 and i > 0):
            print(f"  Trained {i} epochs ...")

    print("Convert QAT model ...")
    quantized_model = quantize_fx.convert_fx(prepared_model.eval())

    with open("qat_quantized.txt", "w") as f:
        print(quantized_model, file=f)

    print("Saving QAT Model ...")
    torch.save(quantized_model.state_dict(), "models/qat_quantized_model.pth")
    print(f"QAT Model Size (MB): {os.path.getsize('models/qat_quantized_model.pth') / 1e6:.2f}")

