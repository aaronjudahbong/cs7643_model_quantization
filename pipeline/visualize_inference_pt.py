
import matplotlib.pyplot as plt
import torch
from src.scripts.mappings import map_train_id_to_color
from src.models.deeplabv3_mnv3 import load_model, get_empty_model
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np

CHECKPOINT = "./models/ptq_int_8_backbone_histogram.pt"

if __name__ == "__main__":
    print("----- Visualizing Inference Results -----")
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    model = torch.jit.load(CHECKPOINT, map_location=device)
    model.eval()

    # Load Image
    sample_folder = "frankfurt"
    sample_id = "000001_014565"
    # sample_folder = "lindau"
    # sample_id = "000047_000019"
    image_path = f"./data/leftImg8bit_trainvaltest/leftImg8bit/val/{sample_folder}/{sample_folder}_{sample_id}_leftImg8bit.png"
    ground_truth_path = f"./data/gtFine_trainIdColorized/gtFine/val/{sample_folder}/{sample_folder}_{sample_id}_gtFine_color.png"
    image = Image.open(image_path).convert("RGB")
    # Normalize image.
    normalize_image = T.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    image_tensor = F.to_tensor(image)
    image_tensor = normalize_image(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, dict):
            output = output["out"]
        predicted_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    # Map train IDs to colors for visualization
    height, width = predicted_mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for train_id, color in map_train_id_to_color.items():
        color_mask[predicted_mask == train_id] = color  

    # Display original image and predicted mask
    plt.figure(constrained_layout=True)
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    ground_truth = Image.open(ground_truth_path)
    plt.imshow(ground_truth)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(color_mask)
    plt.axis("off")
    plt.savefig("./results/quantized_visualization.png")
    plt.close()