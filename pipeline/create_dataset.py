import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F

"""
Code Reference:
https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
"""

class cityScapesDataset(Dataset):
    def __init__(self, image_folder, label_folder, transformations = False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transformations = transformations

        all_images = []
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(".png"):
                    all_images.append(os.path.join(root, file))
        
        all_labels = []
        for root, _, files in os.walk(self.label_folder):
            for file in files:
                if file.lower().endswith(".png"):
                    all_labels.append(os.path.join(root, file))

        #Create paths for each file in the image/label folder
        self.images = list(sorted(all_images))
        self.labels = list(sorted(all_labels))

        assert len(self.images) == len(self.labels)

        #Normalize using the mean/std of ImageNet
        self.normalize_image = T.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RBG")
        label = torch.from_numpy(np.array(Image.open(self.labels[idx]))).long()

        if self.transformations:
            #Apply a horizontal flip
            if random.rand() < 0.5:
                image, label = F.hflip(image), F.hflip(label)

        image = self.normalize_image(image)
        return image, label
