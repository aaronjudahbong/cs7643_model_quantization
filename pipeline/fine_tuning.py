import yaml
from pipeline.create_dataset import cityScapesDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.deeplabv3_mnv3 import get_empty_model, save_model, load_model

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

    training_dataset = cityScapesDataset(training_image_folder, training_label_folder, config['training']['train_transforms'])
    validation_dataset = cityScapesDataset(validation_image_folder, validation_label_folder, config['training']['val_transforms'])

    print("Preparing Training and Validation Dataloader ...")
    training_loader = DataLoader(training_dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=True)

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    model = load_model(get_empty_model(19), "./models/baseline_init_model.pth", device=device)
    model = model.to(device)

    print("Preparing model for fine-tuning ...")
    #Freeze all but the last 3 layers. Only optimize these
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze last 3 classifier layers
    for name, param in model.named_parameters():
        if (
            "classifier.1" in name or   # Conv2d
            "classifier.2" in name or   # BatchNorm2d
            "classifier.4" in name      # Conv2d
        ):
            param.requires_grad = True

    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=float(config['training']['learning_rate']),
                                                weight_decay=float(config['training']['weight_decay']))

    epochs = config['training']['epochs']
    print("Starting Fine-Tuning ...")
    for epoch in range(epochs):
        print(f"--- Epoch {epoch} ---")
        model.train()
        training_loss = 0
        for image, labels in training_loader:
            optimizer.zero_grad()
            out = model(image)['out']
            loss = loss_function(out, labels)
            training_loss += loss
            loss.backward()
            optimizer.step()
        
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for image, labels in validation_loader:
                out = model(image)['out']
                loss = loss_function(out, labels)
                validation_loss += loss
        
        average_training_loss = training_loss / len(training_loader)
        average_validation_loss = validation_loss / len(validation_loader)
        print(f"Epoch: {epoch}, Training Loss: {average_training_loss}, Validation Loss: {average_validation_loss}")
        
