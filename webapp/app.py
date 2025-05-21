from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from src.model import QuickDrawCNN
from src.config import CLASSES
from torchvision import transforms

import cairocffi as cairo
import numpy as np
from rdp import rdp
from scipy.interpolate import interp1d

import os

app = Flask(__name__)

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

model = QuickDrawCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load("models/model_quickdraw_10000.pt", map_location="cpu"))
model.eval()

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

# Function to preprocess strokes, to be used before vector_to_raster
def preprocess_strokes(raw_strokes):
	# 1) Clean and accumulate for bbox
	cleaned = []
	all_pts = []
	for xs, ys in raw_strokes:
		if not isinstance(xs, list) or not isinstance(ys, list):
			continue
		if len(xs) < 2 or len(xs) != len(ys):
			continue
		pts = np.stack([xs, ys], axis=1).astype(np.float32)
		cleaned.append(pts)
		all_pts.append(pts)
	if not all_pts:
		return []

	all_pts = np.vstack(all_pts)

	# 2) Translate so min→0
	min_xy = all_pts.min(axis=0)
	all_pts -= min_xy

	# 3) Uniform scale so max span→255
	span = all_pts.max(axis=0)
	scale = 255.0 / max(span[0], span[1])
	all_pts *= scale

	# 4) Re-split into strokes & process each
	processed = []
	idx = 0
	for original in cleaned:
		n = original.shape[0]
		stroke = all_pts[idx:idx+n]
		idx += n

		# Remove consecutive identical points
		diffs = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
		mask = np.concatenate(([True], diffs > 0))
		stroke = stroke[mask]
		if stroke.shape[0] < 2:
			continue

		# Re-sample with spacing ≈1px
		d = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
		D = np.concatenate(([0.0], np.cumsum(d)))
		L = D[-1]
		if L < 1.0:
			continue
		num = int(np.ceil(L))
		target = np.linspace(0, L, num=num)
		fx = interp1d(D, stroke[:, 0], kind='linear')
		fy = interp1d(D, stroke[:, 1], kind='linear')
		resampled = np.stack([fx(target), fy(target)], axis=1)

		# Simplify with RDP (ε=2.0) (as in the documentation)
		simp = rdp(resampled, epsilon=2.0)
		if len(simp) < 2:
			continue

		processed.append([
			simp[:, 0].astype(np.float32),
			simp[:, 1].astype(np.float32),
		])

	return processed


@app.route("/")
def index():
	return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

	raw_strokes = request.json

	vector_image = preprocess_strokes(raw_strokes)

	raster_list = vector_to_raster([vector_image])

	raster_np = raster_list[0]
	raster_np = raster_np.reshape(28, 28)
	pil_img = Image.fromarray(raster_np).convert("L")

	#os.makedirs("tmp", exist_ok=True)
	#pil_img.save("./tmp/image.png")

	tensor = transform(pil_img).unsqueeze(0)

	with torch.no_grad():
		output = model(tensor)
		probs = torch.softmax(output, dim=1)[0]
		top5 = torch.topk(probs, 5)

		results = [{"label": CLASSES[idx], "prob": float(prob)} for idx, prob in zip(top5.indices, top5.values)]

	return jsonify({"top5": results})


if __name__ == "__main__":
	app.run(debug=True)
