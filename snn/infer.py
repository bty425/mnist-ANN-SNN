import torch
from torchvision import transforms
from PIL import Image
import sys
from model import MNISTNet

def infer(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1, keepdim=True)
            confidence = torch.softmax(output, dim=1).max().item()

        print(f"�KpW: {prediction.item()}")
        print(f"n�: {confidence:.4f}")

        return prediction.item(), confidence

    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python infer.py <image_path>")
    else:
        infer(sys.argv[1])