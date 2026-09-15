"""
Centralized configuration for the Satellite Imagery Analyzer.

Contains dataset paths, model hyperparameters, class labels, and
visualization settings used across all modules.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EUROSAT_DIR = DATA_DIR / "EuroSAT_RGB"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Ensure output directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── EuroSAT Dataset ─────────────────────────────────────────────
# 10 land-use / land-cover classes from Sentinel-2 satellite imagery
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(CLASS_NAMES)

# Class-to-index and index-to-class mappings
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ─── Model Hyperparameters ───────────────────────────────────────
IMAGE_SIZE = 64            # EuroSAT images are 64x64 pixels
BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ImageNet normalization stats (for transfer learning with pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Best model checkpoint filename
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

# ─── NDVI Settings ───────────────────────────────────────────────
# NDVI vegetation classification thresholds
NDVI_THRESHOLDS = {
    "Water":                (-1.0, 0.0),
    "Barren/Built-up":      (0.0, 0.1),
    "Sparse Vegetation":    (0.1, 0.3),
    "Moderate Vegetation":  (0.3, 0.6),
    "Dense Vegetation":     (0.6, 1.0),
}

# Color map for NDVI visualization (category → hex color)
NDVI_COLORS = {
    "Water":                "#1a5276",
    "Barren/Built-up":      "#d4a373",
    "Sparse Vegetation":    "#e9c46a",
    "Moderate Vegetation":  "#81b29a",
    "Dense Vegetation":     "#2d6a4f",
}

# ─── Change Detection ────────────────────────────────────────────
CHANGE_THRESHOLD = 0.15    # Minimum NDVI difference to flag as "changed"

# ─── API Settings ────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
MAX_UPLOAD_SIZE_MB = 10
