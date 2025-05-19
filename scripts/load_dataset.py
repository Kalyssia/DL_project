import os
import urllib.request
import numpy as np

with open("all_classes.txt", "r") as f:
    CLASSES = [line.strip() for line in f]

MAX_CLASSES = 345
CLASSES = CLASSES[:MAX_CLASSES]
URL_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
OUT_DIR = "data_npy"
MAX_IMAGES = 1000

os.makedirs(OUT_DIR, exist_ok=True)

for idx, cls in enumerate(CLASSES, 1):
    print(f"[{idx}/{len(CLASSES)}] Downloading '{cls}'...")

    url = URL_BASE + cls.replace(" ", "%20") + ".npy"
    npy_path = os.path.join(OUT_DIR, f"{cls}.npy")

    try:
        urllib.request.urlretrieve(url, npy_path)
        data = np.load(npy_path)[:MAX_IMAGES]
        data = data.reshape(-1, 28, 28).astype(np.uint8)
        np.save(npy_path, data)
        print(f"Saved {data.shape[0]} samples for '{cls}'")
    except Exception as e:
        print(f"Error with '{cls}': {e}")
