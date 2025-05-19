import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
from config import CLASSES
import os

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
	transforms.Grayscale(),
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder("data", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = SimpleCNN(num_classes=len(CLASSES)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for images, labels in train_loader:
		images, labels = images.to(DEVICE), labels.to(DEVICE)
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		_, predicted = torch.max(outputs, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	acc = 100 * correct / total
	print(f"Epoch {epoch+1} | Loss: {running_loss:.4f} | Acc: {acc:.2f}%")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/simple_cnn.pt")