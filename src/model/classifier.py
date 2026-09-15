import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.config import NUM_CLASSES

class SatelliteClassifier(nn.Module):
    """
    Satellite Classifier based on ResNet-18.
    """
    def __init__(self, pretrained: bool = True):
        super(SatelliteClassifier, self).__init__()
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Replace final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    def freeze_backbone(self):
        """Freezes all convolutional layers."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def get_model(pretrained: bool = True, freeze: bool = True) -> SatelliteClassifier:
    """
    Convenience function to initialize and optionally freeze the model.
    """
    model = SatelliteClassifier(pretrained=pretrained)
    if freeze:
        model.freeze_backbone()
    return model
