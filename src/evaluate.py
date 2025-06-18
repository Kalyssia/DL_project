import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
from model import SimpleCNN
from config import CLASSES

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
    x = x[:, None, :, :] / 255.0
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

print("Loading validation dataset...")
x, y = load_dataset()
dataset = TensorDataset(x, y)
_, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
val_loader = DataLoader(val_dataset, batch_size=64)

model = SimpleCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load("models/simple_cnn.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

total = 0
correct = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(xticks_rotation=45, cmap="Blues")
plt.tight_layout()
plt.show()
