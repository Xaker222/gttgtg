import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import DigitCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(image_path, model_path="weights/digit_cnn.pth"):
    model = DigitCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    return pred.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image with digit")
    args = parser.parse_args()

    digit = predict_image(args.image)
    print(f"Predicted digit: {digit}")
