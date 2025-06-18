import os
import json
import requests
import numpy as np
from PIL import Image
import torch
import cairocffi as cairo
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms

# Configuration
CLASSES_FILE = "all_classes.txt"
URL_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
OUT_DIR = "data_tensors_10000"
N_INSTANCES = 10000      # number of drawings per class
MAX_WORKERS = 16        # number of parallel threads

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# Define your PIL→tensor transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Original function used to convert vector images to raster images, from google's documentation
def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)
    
    return raster_images


def process_class(cls_name, n_instances=N_INSTANCES):
    """
    Download the simplified NDJSON for `cls_name`, convert the first
    n_instances drawings to tensors, and save them as a single .pt file.
    """
    url = URL_BASE + cls_name.replace(" ", "%20") + ".ndjson"
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    tensors = []
    for i, raw in enumerate(resp.iter_lines(decode_unicode=True)):
        if i >= n_instances:
            break
        if not raw:
            continue

        # Parse the JSON line
        entry = json.loads(raw)
        drawing = entry["drawing"]  # list of strokes

        # Build vector_image: list of [x_array, y_array]
        vector_image = [
            [
                np.array(stroke[0], dtype=np.float32),
                np.array(stroke[1], dtype=np.float32)
            ]
            for stroke in drawing
        ]

        # Rasterize and reshape to 28×28
        raster = vector_to_raster([vector_image])[0]
        raster = raster.reshape(28, 28).astype(np.uint8)

        # Convert to PIL and then to tensor
        pil = Image.fromarray(raster).convert("L")
        tensor = transform(pil).unsqueeze(0)  # shape (1, C, 28, 28)

        tensors.append(tensor)

    # Stack all tensors and save
    all_tensors = torch.cat(tensors, dim=0)
    out_path = os.path.join(OUT_DIR, f"{cls_name}.pt")
    torch.save(all_tensors, out_path)


if __name__ == "__main__":
    # Read class names (limit to 345)
    with open(CLASSES_FILE, "r") as f:
        classes = [l.strip() for l in f][:345]

    total = len(classes)
    done = 0

    # Process classes in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_cls = {executor.submit(process_class, cls): cls for cls in classes}
        for future in as_completed(future_to_cls):
            cls = future_to_cls[future]
            done += 1
            try:
                future.result()
                print(f"[{done}/{total}] '{cls}' done.")
            except Exception as e:
                print(f"[{done}/{total}] Error processing '{cls}': {e}")
