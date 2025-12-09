import os
import glob
import yaml
from PIL import Image
from tqdm import tqdm

import numpy as np
import itertools
import json
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F

from src.models.deeplabv3_mnv3 import get_empty_model, load_model
from src.quantization.quantization_utils import set_seed, build_qconfig
from pipeline.create_dataset import cityScapesDataset
from pipeline.metrics import calculate_miou

def plot_loss(train_losses, val_losses):
    fig, ax = plt.subplots()
    epochs = range(1, len(train_losses)+1)
    ax.plot(epochs, train_losses, marker='o', label='train')
    ax.plot(epochs, val_losses, marker='o', label='val')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.set_title('Training and Validation Loss')
    ax.grid()
    return fig, ax

def plot_miou(mious):
    fig, ax = plt.subplots()
    epochs = range(1, len(mious)+1)
    ax.plot(epochs, mious, marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mIOU')
    ax.set_title('Validation mIOU')
    ax.grid()
    return fig, ax

def run_qat(idx, config, results_dir):
    set_seed()
    print("--- Running QAT Script ---")

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    print(f"Loading Model from checkpoint: {config['model_checkpoint']} ...")
    # Get empty model and load checkpoint weights.
    model = get_empty_model()
    model = load_model(model, config["model_checkpoint"], device=device)
    model.eval()
    print(f"Baseline Model Size (MB): {os.path.getsize(config['model_checkpoint']) / 1e6:.2f}")

    # Get Quantization Configuration
    print(f"Building QConfig with mode: {config['mode']}...")
    qconfig_mapping = build_qconfig("qat", config)
    print(f"qconfig_mapping global config: {qconfig_mapping.global_qconfig}")
    print(f"qconfig_mapping weight config: {qconfig_mapping.global_qconfig.weight}")
    print(f"qconfig_mapping activation config: {qconfig_mapping.global_qconfig.activation}")

    train_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
    train_label_path = "data/gtFine_trainId/gtFine/train"
    val_img_path = "data/leftImg8bit_trainvaltest/leftImg8bit/val"
    val_label_path = "data/gtFine_trainId/gtFine/val"

    cal_dataset = cityScapesDataset(train_img_path, train_label_path, config['training']['train_transforms'])
    train_dataset = cityScapesDataset(train_img_path, train_label_path, config['training']['train_transforms'])
    val_dataset = cityScapesDataset(val_img_path, val_label_path, config['training']['val_transforms'])

    cal_dataloader = DataLoader(cal_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    example_image, _ = next(iter(cal_dataloader))
    example_image = example_image.to(device)
    prepared_model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, (example_image,))
    prepared_model = prepared_model.to(device)
    prepared_model.eval()

    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(prepared_model.parameters(), lr=float(config['training']['learning_rate']),
                                                        weight_decay=float(config['training']['weight_decay']))
    scheduler = CosineAnnealingLR(optimizer, T_max = config['training']['epochs'], eta_min = 1e-5)
    
    print("Start Calibration...")
    if config['calibration']['enabled']:
        with torch.no_grad():
            for i, (image, _) in enumerate(cal_dataloader):
                image = image.to(device, non_blocking=True)

                prepared_model(image)
                if (i % 10 == 0 and i > 0):
                    print(f"  Calibrated {i} batches ...")

                if (i >= config['calibration']['steps'] - 1):
                    print(f"  Completed {config['calibration']['steps']} calibration steps.")
                    break


    print("Starting QAT...")
    train_losses = []
    val_losses = []
    val_mious = []

    epochs = config['training']['epochs']
    for epoch in range(epochs):
        prepared_model.train()
        training_loss = 0
        for image, label in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = prepared_model(image)['out']
            loss = loss_function(out, label)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Calculate mIoU on validation set after each batch
            prepared_model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for val_image, val_label in val_dataloader:
                    val_image = val_image.to(device, non_blocking=True)
                    val_label = val_label.to(device, non_blocking=True)
                    val_out = prepared_model(val_image)['out']
                    val_preds = val_out.argmax(dim=1)
                    val_predictions.append(val_preds.cpu())
                    val_targets.append(val_label.cpu())
            
            # Concatenate and calculate mIoU
            val_predictions = torch.cat(val_predictions, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
            val_miou, _ = calculate_miou(val_predictions, val_targets, num_classes=19, ignore_index=255)
            print(f"  Batch {len(train_losses) if train_losses else 0} - Validation mIoU: {val_miou:.4f}")
            
            # Set back to train mode
            prepared_model.train()

        prepared_model.eval()
        validation_loss = 0
        val_miou = 0
        with torch.no_grad():
            for image, label in tqdm(val_dataloader, desc=f"Validation Epoch {epoch}"):
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                out = prepared_model(image)['out']
                pred = out.argmax(dim=1)
                loss = loss_function(out, label)
                validation_loss += loss.item()
                val_miou += calculate_miou(pred, label)[0]
        
        scheduler.step()

        average_training_loss = training_loss / len(train_dataloader)
        average_validation_loss = validation_loss / len(val_dataloader)
        average_val_miou = val_miou / len(val_dataloader)

        train_losses.append(average_training_loss)
        val_losses.append(average_validation_loss)
        val_mious.append(average_val_miou)

        print(f"Epoch: {epoch}, Training Loss: {average_training_loss}, Validation Loss: {average_validation_loss}, mIOU: {average_val_miou}")


    
    # Run inference on full validation set and calculate mIoU
    print(f"\nRunning inference on validation set...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, (image, labels) in enumerate(tqdm(val_dataloader, desc="Validation inference")):
            image = image.to(device, non_blocking=True)
            
            out = prepared_model(image)['out']
            preds = out.argmax(dim=1)
            
            all_predictions.append(preds.cpu())
            all_targets.append(labels)
            
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate mIoU
    print(f"\nCalculating mIoU...")
    miou, per_class_ious = calculate_miou(all_predictions, all_targets, num_classes=19, ignore_index=255)
    print(f"mIoU: {miou:.4f}")

    print("Convert QAT model ...")
    # must move model to CPU to convert, else it errors!
    prepared_model = prepared_model.cpu()
    quantized_model = quantize_fx.convert_fx(prepared_model.eval())
    quantized_model = quantized_model.to(device).eval()
    for i in range(2):
        # Run inference on full validation set and calculate mIoU
        print(f"\nRunning inference on validation set AFTER...")
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for i, (image, labels) in enumerate(tqdm(val_dataloader, desc="Validation inference")):
                image = image.to(device, non_blocking=True)
                
                out = quantized_model(image)['out']
                preds = out.argmax(dim=1)
                
                all_predictions.append(preds.cpu())
                all_targets.append(labels)
                
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate mIoU
        print(f"\nCalculating mIoU...")
        miou, per_class_ious = calculate_miou(all_predictions, all_targets, num_classes=19, ignore_index=255)
        print(f"mIoU: {miou:.4f}")

    print("Saving QAT Model ...")
    model_path = os.path.join(results_dir, f"qat_quantized_model_{idx}.pth")
    torch.save(quantized_model.state_dict(), model_path)
    print(f"QAT Model Size (MB): {os.path.getsize(model_path) / 1e6:.2f}")

    # save all results
    result = {
      "idx": idx,
      "config": config,
      "train_losses": [round(loss, 2) for loss in train_losses],
      "val_losses": [round(loss, 2) for loss in val_losses],
      "val_mious": [round(miou, 2) for miou in val_mious],
      "final_train_loss": train_losses[-1],
      "final_val_loss": val_losses[-1],
      "final_val_miou": miou
    }

    json_path = os.path.join(results_dir, "qat_results.json")
    if os.path.exists(json_path):
      with open(json_path, "r") as f:
        results = json.load(f)
    else:
      results = []

    results.append(result)

    with open(json_path, "w") as f:
      json.dump(results, f, indent=2)

    # save all results
    fig, ax = plot_loss(train_losses, val_losses)
    plot_path = os.path.join(results_dir, f"qat_loss_{idx}.png")
    fig.savefig(plot_path)

    fig, ax = plot_miou(val_mious)
    plot_path = os.path.join(results_dir, f"miou_{idx}.png")
    fig.savefig(plot_path)

    del prepared_model
    del quantized_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    results_dir = "qat_results"
    os.makedirs(results_dir, exist_ok=True)

    # Load configuration.
    print("Loading Configuration ...")
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)["qat"]

    run_qat(0, config, results_dir)
