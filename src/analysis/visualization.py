import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.figure import Figure
from collections import Counter

def plot_ndvi_map(ndvi_array: np.ndarray, title: str = 'NDVI Map', save_path: str = None) -> Figure:
    """
    Color-coded NDVI heatmap using RdYlGn colormap, with colorbar from -1 to 1.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cax = ax.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label='NDVI')
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig

def plot_confusion_matrix(y_true: list, y_pred: list, class_names: list, save_path: str = None) -> Figure:
    """
    Publication-quality confusion matrix with sklearn.metrics.confusion_matrix, 
    using seaborn heatmap with annotations.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig

def plot_training_curves(history: dict, save_path: str = None) -> Figure:
    """
    2-subplot figure: loss curves (train vs val) and accuracy curves (train vs val) over epochs.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history.get('loss', [])) + 1)
    
    ax1.plot(epochs, history.get('loss', []), label='Train Loss')
    ax1.plot(epochs, history.get('val_loss', []), label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    ax2.plot(epochs, history.get('accuracy', []), label='Train Accuracy')
    ax2.plot(epochs, history.get('val_accuracy', []), label='Val Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig

def plot_change_detection(ndvi_before: np.ndarray, ndvi_after: np.ndarray, difference_map: np.ndarray, save_path: str = None) -> Figure:
    """
    3-panel side-by-side figure: Before NDVI, After NDVI, Change Map (with gain=green, loss=red colormap).
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(ndvi_before, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0].set_title('Before NDVI')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')
    
    im1 = axes[1].imshow(ndvi_after, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title('After NDVI')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')
    
    im2 = axes[2].imshow(difference_map, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[2].set_title('Change Map (Gain=Green, Loss=Red)')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig

def plot_class_distribution(predictions: list[dict], save_path: str = None) -> Figure:
    """
    Horizontal bar chart of predicted class counts.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    counts = Counter([p.get('class_name', 'Unknown') for p in predictions])
    classes = list(counts.keys())
    freqs = list(counts.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(classes, freqs, color='skyblue')
    ax.set_xlabel('Count')
    ax.set_title('Predicted Class Distribution')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig
