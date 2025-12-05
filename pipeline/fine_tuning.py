import yaml
from pipeline.create_dataset import cityScapesDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
from torch.utils.data import DataLoader
from src.models.deeplabv3_mnv3 import get_empty_model, save_model, load_model
from tqdm import tqdm
from pipeline.metrics import calculate_miou
import matplotlib.pyplot as plt

training_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/train"
training_label_folder = "./data/gtFine_trainId/gtFine/train"
validation_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/val"
validation_label_folder = "./data/gtFine_trainId/gtFine/val"

if __name__ == "__main__":
    # Load configuration.
    print("----- Running FP Training Script -----")
    with open("configs/ptq.yaml", "r") as f:
        config = yaml.safe_load(f)["fp"]

    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']

    training_dataset = cityScapesDataset(training_image_folder, training_label_folder, config['training']['train_transforms'])
    validation_dataset = cityScapesDataset(validation_image_folder, validation_label_folder, config['training']['val_transforms'])

    print("Preparing Training and Validation Dataloader ...")
    training_loader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    model = load_model(get_empty_model(19), "./models/baseline_init_model.pth", device=device)
    model = model.to(device)

    print("Preparing model for fine-tuning ...")

    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=float(config['training']['learning_rate']),
                                                weight_decay=float(config['training']['weight_decay']))
    
    # scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-5)
    scheduler = PolynomialLR(optimizer, total_iters=epochs, power=0.9)

    best_validation_loss = float('inf')
    print("Starting Fine-Tuning ...")
    performance = torch.zeros((epochs, 3))
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

        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for image, labels in tqdm(validation_loader, desc=f"Validation Epoch {epoch}"):
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                out = model(image)['out']
                loss = loss_function(out, labels)
                validation_loss += loss.item()

                prediction = torch.argmax(out, dim = 1)
                all_predictions.append(prediction.cpu())
                all_targets.append(labels.cpu())

        scheduler.step()
        
        average_training_loss = training_loss / len(training_loader)
        average_validation_loss = validation_loss / len(validation_loader)

        all_predictions = torch.cat(all_predictions, dim = 0)
        all_targets = torch.cat(all_targets, dim = 0)

        miou = calculate_miou(all_predictions, all_targets)

        print(f"Epoch: {epoch}, Training Loss: {average_training_loss}, Validation Loss: {average_validation_loss}, MIOU: {miou}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        performance[epoch, 0], performance[epoch, 1], performance[epoch, 2] = average_training_loss, average_validation_loss, miou

        # Always save the latest model.
        save_model(model, f"./models/finetuned_model_last_epoch.pth")

        # Log training progress to a text file (append)
        with open("./models/finetuned_model_progress.txt", "a") as f:
            f.write(
                f"Epoch {epoch}: Training Loss={average_training_loss:.6f}, "
                f"Validation Loss={average_validation_loss:.6f}\n"
            )

    #Produce Learning Curves
    plt.plot(list(range(0, epochs, 1)), performance[:, 0].cpu(), label="Training Loss")
    plt.plot(list(range(0, epochs, 1)), performance[:, 1].cpu(), label="Validation Loss")
    plt.title("Learning Curve - Training/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid()
    plt.legend()
    plt.savefig("./results/Finetuned_Training_Validation_Curve.png")
    plt.close()

    plt.plot(list(range(0, epochs, 1)), performance[:, 0].cpu(), label="Training Loss")
    plt.plot(list(range(0, epochs, 1)), performance[:, 2].cpu(), label="mIoU")
    plt.title("Learning Curve - Training Loss & mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("Average Quantity")
    plt.grid()
    plt.legend()
    plt.savefig("./results/Finetuned_Training_mIoU_Curve.png")
    plt.close()
