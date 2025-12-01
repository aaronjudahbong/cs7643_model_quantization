import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import glob
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
"""
Code Reference:
https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomCrop.html
"""

class cityScapesDataset(Dataset):
    def __init__(self, image_folder, label_folder, transforms_config):
        self.to_crop = transforms_config.get('crop', False)
        self.to_resize = transforms_config.get('resize', False)
        self.to_flip = transforms_config.get('flip', False)
        print(f"Desired transformations: crop:{self.to_crop}, resize:{self.to_resize}, flip:{self.to_flip}")

        all_images = glob.glob(os.path.join(image_folder, "**", "*.png"), recursive=True)
        all_labels = glob.glob(os.path.join(label_folder, "**", "*.png"), recursive=True)

        #Create paths for each file in the image/label folder
        self.images = list(sorted(all_images))
        self.labels = list(sorted(all_labels))

        assert len(self.images) == len(self.labels)

        # Normalize using the mean/std of ImageNet
        self.normalize_image = T.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx])

        # Apply random scaling to image and mask
        if self.to_resize:
            if random.random() < 0.5:
                # scaling shall be in the range (https://arxiv.org/pdf/1706.05587):
                scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
                # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
                scale = np.random.choice(scales)
                width, height = int(scale*image.size[0]), int(scale*image.size[1])
                image = F.resize(image, size=(height, width), interpolation=F.InterpolationMode.BILINEAR)
                label = F.resize(label, size=(height, width), interpolation=F.InterpolationMode.NEAREST)

        # Apply random crop to image and mask
        if self.to_crop:
            # Pad image and mask if needed
            crop_size = 769
            pad_width = max(0, crop_size - image.size[0])//2+1
            pad_height = max(0, crop_size - image.size[1])//2+1
            image = F.pad(image, padding=(pad_width, pad_height, pad_width, pad_height))
            label = F.pad(label, padding=(pad_width, pad_height, pad_width, pad_height), fill=255)

            row, col, height, width = T.RandomCrop.get_params(image, (crop_size,crop_size))
            image = F.crop(image, row, col, height, width)
            label = F.crop(label, row, col, height, width)

        # Apply a horizontal flip to image and mask
        if self.to_flip:
            if random.random() < 0.5:
                image, label = F.hflip(image), F.hflip(label)

        label = torch.from_numpy(np.array(label)).long()
        image = self.normalize_image(F.to_tensor(image))

        return image, label
