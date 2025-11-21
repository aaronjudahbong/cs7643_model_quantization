import torch
import torch.nn as nn
import torchvision

# We will use the DeepSeekV3 with MobileNetV3 backbone model.
# # https://docs.pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html#deeplabv3_mobilenet_v3_large

# We will use the same taxonomy as the official CityScapes evaluation taxonomy. NUM_CLASSES = 19
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
NUM_CLASSES = 19

def get_empty_model(num_classes = NUM_CLASSES):
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=None,
        num_classes=num_classes,
    )

    # Replace classifier head
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def get_pretrained_model(num_classes = NUM_CLASSES):
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights="COCO_WITH_VOC_LABELS_V1"
    )

    # Replace classifier head
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model