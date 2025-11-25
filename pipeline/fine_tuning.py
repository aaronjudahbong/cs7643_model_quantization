from create_dataset import cityScapesDataset
import torch
from torch.utils.data import DataLoader

training_image_folder = "../data/leftImg8bit_trainvaltest/leftImg8bit/train"
training_label_folder = "../data/gtFine_trainId/gtFine/train"
validation_image_folder = "../data/leftImg8bit_trainvaltest/leftImg8bit/val"
validation_label_folder = "../data/gtFine_trainId/gtFine/val"
batch_size = 128

training_dataset = cityScapesDataset(training_image_folder, training_label_folder)
validation_dataset = cityScapesDataset(validation_image_folder, validation_label_folder)

training_loader = DataLoader(training_dataset, batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size, shuffle=True)
