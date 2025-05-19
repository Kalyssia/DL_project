import os
import urllib.request
import numpy as np
from PIL import Image

CLASSES = ["cat", "dog", "car", "tree", "bicycle", "airplane", "clock", "apple", "face", "house"]
URL_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
OUT_DIR = "data"
MAX_IMAGES = 10000

os.makedirs(OUT_DIR, exist_ok=True)

# For each class, download the corresponding images, reshape it, and save in data/<class_name>/
for cls in CLASSES:
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
