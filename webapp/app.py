from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image, ImageChops, ImageOps
import base64
import io

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import MLP
from src.config import CLASSES
from torchvision import transforms



app = Flask(__name__)

transform = transforms.Compose([
	transforms.Grayscale(),
	transforms.Resize((28, 28)),
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

model = MLP(num_classes=len(CLASSES))
model.load_state_dict(torch.load("models/mlp.pt", map_location="cpu"))
model.eval()

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
	data = request.json["image"]
	header, encoded = data.split(",", 1)
	img_bytes = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
	image = autocrop(image)
	image = ImageOps.invert(image.convert("L")).convert("RGB")
	tensor = transform(image).unsqueeze(0)
	with torch.no_grad():
		output = model(tensor)
		_, pred = torch.max(output, 1)
	return jsonify({"prediction": CLASSES[pred.item()]})

def autocrop(pil_img, bgcolor="white"):
	bg = Image.new(pil_img.mode, pil_img.size, bgcolor)
	diff = ImageChops.difference(pil_img, bg)
	bbox = diff.getbbox()
	if bbox:
		cropped = pil_img.crop(bbox)
		side = max(cropped.size)
		square = Image.new("RGB", (side, side), bgcolor)
		square.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
		return square
	else:
		return pil_img

if __name__ == "__main__":
	app.run(debug=True)
