import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, 3)
		self.pool = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.fc1 = nn.Linear(64 * 5 * 5, 128)
		self.fc2 = nn.Linear(128, num_classes)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 64 * 5 * 5)
		x = F.relu(self.fc1(x))
		return self.fc2(x)

class BiggerCNN(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

		with torch.no_grad():
			dummy = torch.zeros(1, 1, 28, 28)
			out = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy)))))
			out = self.pool(F.relu(self.conv4(F.relu(self.conv3(out)))))
			self.flattened_size = out.view(1, -1).size(1)

		self.fc1 = nn.Linear(self.flattened_size, 256)
		self.dropout = nn.Dropout(0.5)
		self.fc2 = nn.Linear(256, num_classes)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool(F.relu(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = self.pool(F.relu(self.conv4(x)))
		x = x.view(x.size(0), -1)
		x = self.dropout(F.relu(self.fc1(x)))
		return self.fc2(x)


class MLP(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.fc1 = nn.Linear(28 * 28, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, num_classes)
		self.dropout = nn.Dropout(0.3)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		return self.fc3(x)
