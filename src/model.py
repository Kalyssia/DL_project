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


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, downsample=False):
		super().__init__()
		stride = 2 if downsample else 1
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
			stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
			stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample
		if downsample or in_channels != out_channels:
			self.residual = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1,
					stride=stride, bias=False),
				nn.BatchNorm2d(out_channels)
			)
		else:
			self.residual = nn.Identity()

	def forward(self, x):
		identity = self.residual(x)
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += identity
		return F.relu(out)

class QuickDrawCNN(nn.Module):
	def __init__(self, num_classes=345):
		super().__init__()
		# Stem
		self.stem = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True)
		)
		# Bloques residuales
		self.layer1 = self._make_layer(64, 128, num_blocks=2, downsample=True)
		self.layer2 = self._make_layer(128, 256, num_blocks=2, downsample=True)
		self.layer3 = self._make_layer(256, 512, num_blocks=2, downsample=True)
		# Clasificador
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.dropout = nn.Dropout(p=0.5)
		self.fc = nn.Linear(512, num_classes)

		# Inicialización
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, in_c, out_c, num_blocks, downsample):
		layers = []
		layers.append(ResidualBlock(in_c, out_c, downsample=downsample))
		for _ in range(1, num_blocks):
			layers.append(ResidualBlock(out_c, out_c))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.stem(x) # [N, 64, 28, 28]
		x = self.layer1(x) # [N,128,14,14]
		x = self.layer2(x) # [N,256,7,7]
		x = self.layer3(x) # [N,512,4,4] (gracias al padding)
		x = self.avgpool(x) # [N,512,1,1]
		x = torch.flatten(x, 1) # [N,512]
		x = self.dropout(x)
		logits = self.fc(x) # [N,345]
		return logits
