from pathlib import Path

SEED = 42

DATA_DIR = Path("data")
MASKS_DIR = Path("data_masks")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

CLASSES = ["Guz", "Zwapnienie"]
CLASS_TO_IDX = {"Guz": 0, "Zwapnienie": 1}

MAMMONET = dict(size=128, channels=1, batch_size=32, lr=5e-4, epochs=20)
RESNET = dict(size=224, channels=3, batch_size=32, lr=1e-4, epochs=15)

NORM_GRAY = ([0.5], [0.5])
NORM_IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
