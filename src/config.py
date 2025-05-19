import os

with open("all_classes.txt", "r") as f:
    CLASSES = [line.strip() for line in f]

MAX_CLASSES = 345
CLASSES = CLASSES[:MAX_CLASSES]
