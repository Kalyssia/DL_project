import torch
from torchvision import transforms
from PIL import Image
import sys
from model import SimpleCNN
from config import CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
	transforms.Grayscale(),
	transforms.Resize((28, 28)),
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

model = SimpleCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load("models/simple_cnn.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
	outputs = model(image)
	_, predicted = torch.max(outputs, 1)
	print(f"Prediction: {CLASSES[predicted.item()]}")
