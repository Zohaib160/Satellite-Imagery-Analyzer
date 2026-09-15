import numpy as np
from src.config import NDVI_THRESHOLDS

def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """
    Computes NDVI = (NIR - Red) / (NIR + Red).
    Handles division by zero by returning 0 where the denominator is 0.
    """
    numerator = nir_band - red_band
    denominator = nir_band + red_band
    
    ndvi = np.where(denominator == 0, 0.0, numerator / denominator)
    return ndvi

def classify_vegetation(ndvi_array: np.ndarray) -> np.ndarray:
    """
    Classifies each pixel into categories using NDVI_THRESHOLDS from config.
    Returns a string array of class names.
    """
    classified = np.empty(ndvi_array.shape, dtype=object)
    
    for category, (low, high) in NDVI_THRESHOLDS.items():
        if category == "Dense Vegetation":
            mask = (ndvi_array >= low) & (ndvi_array <= high)
        else:
            mask = (ndvi_array >= low) & (ndvi_array < high)
        classified[mask] = category
        
    return classified

def compute_statistics(ndvi_array: np.ndarray) -> dict:
    """
    Returns min, max, mean, std, and percentage of each vegetation class.
    """
    classified = classify_vegetation(ndvi_array)
    total_pixels = ndvi_array.size
    
    stats = {
        "min": float(np.min(ndvi_array)),
        "max": float(np.max(ndvi_array)),
        "mean": float(np.mean(ndvi_array)),
        "std": float(np.std(ndvi_array)),
        "class_percentages": {}
    }
    
    for category in NDVI_THRESHOLDS.keys():
        count = np.sum(classified == category)
        stats["class_percentages"][category] = float(count / total_pixels) * 100.0
        
    return stats

def generate_sample_data(height: int = 256, width: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates synthetic NIR and Red band arrays with realistic patterns 
    (gradient with noise to simulate a landscape with water, vegetation, and urban areas)
    for demo purposes.
    """
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    base = xx + yy
    
    nir = 0.5 * base + np.random.normal(0, 0.05, (height, width))
    red = 0.5 * (2 - base) + np.random.normal(0, 0.05, (height, width))
    
    # Simulate water
    water_mask = (xx - 0.2)**2 + (yy - 0.2)**2 < 0.1
    nir[water_mask] = 0.1
    red[water_mask] = 0.4
    
    # Simulate vegetation
    veg_mask = (xx - 0.8)**2 + (yy - 0.8)**2 < 0.15
    nir[veg_mask] = 0.8
    red[veg_mask] = 0.1
    
    return np.clip(nir, 0, 1), np.clip(red, 0, 1)
