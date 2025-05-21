import os

DATA_DIR = "data_tensors_10000"
OUTPUT_FILE = "all_classes_from_data.txt"

# Lister tous les fichiers .pt et enlever l'extension
classes = [f[:-3] for f in os.listdir(DATA_DIR) if f.endswith(".pt")]
classes.sort()  # pour garantir un ordre stable

# Écriture dans le fichier
with open(OUTPUT_FILE, "w") as f:
    for cls in classes:
        f.write(cls + "\n")

print(f"{len(classes)} classes écrites dans {OUTPUT_FILE}")
