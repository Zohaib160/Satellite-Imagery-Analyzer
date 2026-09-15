import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from typing import Dict, Any, List
from sklearn.metrics import classification_report

from src.config import BEST_MODEL_PATH, EUROSAT_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, TRAIN_RATIO, VAL_RATIO, CLASS_NAMES
from src.model.dataset import get_data_loaders
from src.model.classifier import get_model

def train_model(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    num_epochs: int, 
    lr: float, 
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Trains the satellite classifier model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Early stopping / Best model saving
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
        print()

    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    
    # Save the best model
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"Model saved to {BEST_MODEL_PATH}")
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Train Satellite Classifier")
    parser.add_argument('--data-dir', type=str, default=str(EUROSAT_DIR), help='Path to EuroSAT data')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir, args.batch_size, TRAIN_RATIO, VAL_RATIO
    )

    # Create model
    model = get_model(pretrained=True, freeze=True)
    model = model.to(device)

    # Train
    history = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Test Set Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

if __name__ == '__main__':
    main()
