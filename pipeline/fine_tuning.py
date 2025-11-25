from pipeline.create_dataset import cityScapesDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.deeplabv3_mnv3 import get_pretrained_model, save_model, load_model

training_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/train"
training_label_folder = "./data/gtFine_trainId/gtFine/train"
validation_image_folder = "./data/leftImg8bit_trainvaltest/leftImg8bit/val"
validation_label_folder = "./data/gtFine_trainId/gtFine/val"
batch_size = 2

training_dataset = cityScapesDataset(training_image_folder, training_label_folder)
validation_dataset = cityScapesDataset(validation_image_folder, validation_label_folder)

training_loader = DataLoader(training_dataset, batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size, shuffle=True)

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = load_model(get_pretrained_model(19), "./models/baseline_init_model.pth", device=device)
loss_function = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

epochs = 10
for epoch in range(epochs):
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
    
