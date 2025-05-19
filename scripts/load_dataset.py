import os
import urllib.request
import numpy as np
from PIL import Image

with open("all_classes.txt", "r") as f:
    CLASSES = [line.strip() for line in f]

MAX_CLASSES = 345  # total of 345 classes
CLASSES = CLASSES[:MAX_CLASSES]
URL_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
OUT_DIR = "data"
MAX_IMAGES = 1000

os.makedirs(OUT_DIR, exist_ok=True)

# For each class, download the corresponding images, reshape it, and save in data/<class_name>/
for idx, cls in enumerate(CLASSES, 1):
    print(f"[{idx}/{len(CLASSES)}] Downloading '{cls}'...")
    
    url = URL_BASE + cls.replace(" ", "%20") + ".npy"
    npy_path = os.path.join(OUT_DIR, cls + ".npy")
    urllib.request.urlretrieve(url, npy_path)
    data = np.load(npy_path)
    data = data.reshape(-1, 28, 28).astype(np.uint8)
    class_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(class_dir, exist_ok=True)
    for i in range(MAX_IMAGES):
        img = Image.fromarray(data[i])
        img.save(os.path.join(class_dir, f"{i}.png"))
    os.remove(npy_path)
