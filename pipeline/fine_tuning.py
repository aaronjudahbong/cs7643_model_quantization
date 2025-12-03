import yaml
from pipeline.create_dataset import cityScapesDataset
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.deeplabv3_mnv3 import get_empty_model, save_model, load_model
from tqdm import tqdm

training_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/train"
training_label_folder = "./data/gtFine_trainId/gtFine/train"
validation_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/val"
validation_label_folder = "./data/gtFine_trainId/gtFine/val"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Load configuration.
    print("----- Running FP Training Script -----")
    set_seed()
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)["fp"]

    batch_size = config['training']['batch_size']

    training_dataset = cityScapesDataset(training_image_folder, training_label_folder, config['training']['train_transforms'])
    validation_dataset = cityScapesDataset(validation_image_folder, validation_label_folder, config['training']['val_transforms'])

    print("Preparing Training and Validation Dataloader ...")
    training_loader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    model = load_model(get_empty_model(19), "./models/baseline_init_model.pth", device=device)
    model = model.to(device)

    print("Preparing model for fine-tuning ...")
    #Freeze all but the last 3 layers. Only optimize these
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze ASPP Layers. 
    aspp_params = []
    for name, param in model.named_parameters():
        if ("classifier.0.convs.1" in name or
            "classifier.0.convs.2" in name or
            "classifier.0.convs.3" in name or
            "classifier.0.project" in name):
            param.requires_grad = True
            aspp_params.append(param)

    # Unfreeze Classifier Layers.
    classifier_params = []
    for name, param in model.named_parameters():
        if (
            "classifier.1" in name or   # Conv2d
            "classifier.2" in name or   # BatchNorm2d
            "classifier.4" in name      # Conv2d
        ):
            param.requires_grad = True
            classifier_params.append(param)

    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam([
        {"params": aspp_params, "lr": float(config['training']['aspp_learning_rate'])},
        {"params": classifier_params, "lr": float(config['training']['learning_rate'])}
    ], weight_decay=float(config['training']['weight_decay']))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer
    )


    epochs = config['training']['epochs']
    best_validation_loss = float('inf')
    print("Starting Fine-Tuning ...")
    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for image, labels in tqdm(training_loader, desc=f"Training Epoch {epoch}"):
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(image)['out']
            loss = loss_function(out, labels)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for image, labels in tqdm(validation_loader, desc=f"Validation Epoch {epoch}"):
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                out = model(image)['out']
                loss = loss_function(out, labels)
                validation_loss += loss.item()
        
        average_training_loss = training_loss / len(training_loader)
        average_validation_loss = validation_loss / len(validation_loader)

        scheduler.step(average_validation_loss)
        print(f"Epoch: {epoch}, Training Loss: {average_training_loss}, Validation Loss: {average_validation_loss}")

        # Always save the latest model.
        save_model(model, f"./models/finetuned_model_last_epoch.pth")

        # Log training progress to a text file (append)
        with open("./models/finetuned_model_progress.txt", "a") as f:
            f.write(
                f"Epoch {epoch}: Training Loss={average_training_loss:.6f}, "
                f"Validation Loss={average_validation_loss:.6f}\n"
            )

        # Save model if validation loss improved
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            save_model(model, f"./models/finetuned_model_best_epoch_{epoch:03d}.pth")