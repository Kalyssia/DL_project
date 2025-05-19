import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import BiggerCNN
from config import CLASSES
import numpy as np
import os
import multiprocessing

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data_npy"

def load_dataset():
	data = []
	labels = []
	for idx, cls in enumerate(CLASSES):
		class_path = os.path.join(DATA_DIR, f"{cls}.npy")
		if os.path.exists(class_path):
			samples = np.load(class_path)
			data.append(samples)
			labels.append(np.full(len(samples), idx))
	x = np.concatenate(data, axis=0)
	y = np.concatenate(labels, axis=0)
	x = x[:, None, :, :] / 255.0 # reshape to (N, 1, 28, 28) and normalize
	return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def main():
	print(f"Using device: {DEVICE}")
	print("Loading .npy dataset...")

	x, y = load_dataset()
	dataset = TensorDataset(x, y)

	train_size = int(0.8 * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	num_workers = multiprocessing.cpu_count() // 2
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)

	print(f"{len(CLASSES)} classes detected.")
	print("Initializing model...")

	model = BiggerCNN(num_classes=len(CLASSES)).to(DEVICE)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LR)

	print(f"Model on device: {next(model.parameters()).device}")
	print("Starting training...")

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
	torch.save(model.state_dict(), "models/simple_cnn_all_classes.pt")

if __name__ == "__main__":
	main()
