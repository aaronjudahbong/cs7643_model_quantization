import os, glob
import numpy as np
from PIL import Image
from mappings import map_id_to_train_id, map_train_id_to_color

os.makedirs("data/gtFine_trainId/gtFine/", exist_ok=True)
os.makedirs("data/gtFine_trainIdColorized/gtFine/", exist_ok=True)

def remap_dataset_split(split):
    cities = set()
    pattern = f"data/gtFine_trainvaltest/gtFine/{split}/**/*_labelIds.png"
    for path in glob.glob(pattern, recursive=True):
        city = os.path.basename(os.path.dirname(path))
        if city not in cities:
            print(f"  Processing city: {city}")
            cities.add(city)
        img = np.array(Image.open(path))
        remapped = img.copy()
        for k, v in map_id_to_train_id.items():
            remapped[img == k] = v

        out_path = path.replace("data/gtFine_trainvaltest/", "data/gtFine_trainId/").replace("_labelIds.png", "_trainIds.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(remapped.astype(np.uint8)).save(out_path)

def generate_colorized_split(split):
    cities = set()
    pattern = f"data/gtFine_trainId/gtFine/{split}/**/*_trainIds.png"
    for path in glob.glob(pattern, recursive=True):
        city = os.path.basename(os.path.dirname(path))
        if city not in cities:
            print(f"  Processing city: {city}")
            cities.add(city)

        img = np.array(Image.open(path))
        colorized = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for k, v in map_train_id_to_color.items():
            colorized[img == k] = v

        out_path = path.replace("data/gtFine_trainId/gtFine/", "data/gtFine_trainIdColorized/gtFine/").replace("_trainIds.png", "_color.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(colorized.astype(np.uint8)).save(out_path)

if __name__ == "__main__":
    # Remap files in Training and Validation sets.
    print("Remapping Training Split")
    remap_dataset_split("train")
    print("Remapping Training Split Done")
    print("Generating Colorized Training Split")
    generate_colorized_split("train")
    print("Generated Colorized Training Split")
    
    print("Remapping Validation Split")
    remap_dataset_split("val")
    print("Remapping Validation Split Done")
    print("Generating Colorized Validation Split")
    generate_colorized_split("val")
    print("Generated Colorized Validation Split")