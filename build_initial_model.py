from src.models.deeplabv3_mnv3 import get_pretrained_model, save_model

if __name__ == "__main__":
    model = get_pretrained_model()
    save_model(model, "models/baseline_init_model.pth")
    print("Saved initial baseline model to models/baseline_init_model.pth")