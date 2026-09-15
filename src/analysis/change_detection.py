import numpy as np
from src.config import CHANGE_THRESHOLD

def detect_changes(ndvi_before: np.ndarray, ndvi_after: np.ndarray, threshold: float = None) -> dict:
    """
    Computes the difference map (after - before), applies threshold from config to classify 
    pixels as 'gain', 'loss', or 'no change'.
    """
    if threshold is None:
        threshold = CHANGE_THRESHOLD
        
    diff_map = ndvi_after - ndvi_before
    
    change_mask = np.zeros(diff_map.shape, dtype=int)
    change_mask[diff_map > threshold] = 1   # Gain
    change_mask[diff_map < -threshold] = -1 # Loss
    
    total_pixels = diff_map.size
    gain_pixels = np.sum(change_mask == 1)
    loss_pixels = np.sum(change_mask == -1)
    changed_pixels = gain_pixels + loss_pixels
    
    stats = {
        'total_pixels': int(total_pixels),
        'changed_pixels': int(changed_pixels),
        'gain_pixels': int(gain_pixels),
        'loss_pixels': int(loss_pixels),
        'percent_changed': float(changed_pixels / total_pixels * 100),
        'percent_gain': float(gain_pixels / total_pixels * 100),
        'percent_loss': float(loss_pixels / total_pixels * 100),
        'mean_change': float(np.mean(diff_map))
    }
    
    return {
        'difference_map': diff_map,
        'change_mask': change_mask,
        'statistics': stats
    }

def generate_change_report(change_result: dict) -> str:
    """
    Generates a human-readable text report summarizing the change detection results.
    """
    stats = change_result['statistics']
    report = (
        "Change Detection Summary\n"
        "========================\n"
        f"Total Pixels Analyzed: {stats['total_pixels']}\n"
        f"Total Changed Pixels: {stats['changed_pixels']} ({stats['percent_changed']:.2f}%)\n"
        f"  - Vegetation Gain: {stats['gain_pixels']} ({stats['percent_gain']:.2f}%)\n"
        f"  - Vegetation Loss: {stats['loss_pixels']} ({stats['percent_loss']:.2f}%)\n"
        f"Mean NDVI Change: {stats['mean_change']:.4f}\n"
    )
    return report

def generate_sample_change_data(height: int = 256, width: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a pair of synthetic NDVI arrays where the second has simulated deforestation 
    (a rectangular region with reduced NDVI) and urban growth for demo.
    """
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    ndvi_before = 0.5 * (xx + yy) + np.random.normal(0, 0.05, (height, width))
    ndvi_after = np.copy(ndvi_before)
    
    # Simulate deforestation
    ndvi_after[height//4:height//2, width//4:width//2] -= 0.4
    
    # Simulate urban growth
    ndvi_after[height//2:3*height//4, width//2:3*width//4] -= 0.3
    
    return np.clip(ndvi_before, -1, 1), np.clip(ndvi_after, -1, 1)
