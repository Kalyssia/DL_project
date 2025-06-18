import os

with open("all_classes_from_data.txt", "r") as f:
    CLASSES = [line.strip() for line in f]

MAX_CLASSES = 345
CLASSES = CLASSES[:MAX_CLASSES]
