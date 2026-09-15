"""
Tests for the ML model components.

Tests model architecture, prediction output format, and dataset loading.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from src.config import NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class TestSatelliteClassifier:
    """Tests for the SatelliteClassifier model."""

    def test_model_creation(self):
        """Test that the model can be instantiated."""
        from src.model.classifier import SatelliteClassifier
        model = SatelliteClassifier(pretrained=False)
        assert model is not None

    def test_output_shape(self):
        """Test that the model outputs correct number of classes."""
        from src.model.classifier import SatelliteClassifier
        model = SatelliteClassifier(pretrained=False)
        model.eval()

        # Create a dummy input: batch of 4 images, 3 channels, 64x64
        dummy_input = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (4, NUM_CLASSES), (
            f"Expected output shape (4, {NUM_CLASSES}), got {output.shape}"
        )

    def test_output_is_logits(self):
        """Test that raw output contains logits (not probabilities)."""
        from src.model.classifier import SatelliteClassifier
        model = SatelliteClassifier(pretrained=False)
        model.eval()

        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            output = model(dummy_input)

        # Logits can be negative and don't sum to 1
        # Just verify it's a valid tensor with correct shape
        assert output.shape[1] == NUM_CLASSES
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_freeze_backbone(self):
        """Test that freezing the backbone stops gradient computation for conv layers."""
        from src.model.classifier import SatelliteClassifier
        model = SatelliteClassifier(pretrained=False)
        model.freeze_backbone()

        # Check that backbone parameters are frozen
        for name, param in model.backbone.named_parameters():
            if "fc" not in name:
                assert not param.requires_grad, (
                    f"Parameter {name} should be frozen but requires_grad=True"
                )

    def test_unfreeze_backbone(self):
        """Test that unfreezing restores gradient computation."""
        from src.model.classifier import SatelliteClassifier
        model = SatelliteClassifier(pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone()

        for name, param in model.backbone.named_parameters():
            assert param.requires_grad, (
                f"Parameter {name} should be unfrozen but requires_grad=False"
            )

    def test_get_model_convenience(self):
        """Test the get_model convenience function."""
        from src.model.classifier import get_model
        model = get_model(pretrained=False, freeze=True)
        assert model is not None

    def test_single_image_forward(self):
        """Test forward pass with a single image."""
        from src.model.classifier import SatelliteClassifier
        model = SatelliteClassifier(pretrained=False)
        model.eval()

        single_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            output = model(single_image)

        assert output.shape == (1, NUM_CLASSES)


class TestPrediction:
    """Tests for the prediction module."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample image for testing."""
        img_array = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_satellite.jpg"
        img.save(img_path)
        return str(img_path)

    @pytest.fixture
    def sample_image_dir(self, tmp_path):
        """Create a directory with sample images for batch testing."""
        img_dir = tmp_path / "batch_images"
        img_dir.mkdir()
        for i in range(5):
            img_array = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(img_dir / f"image_{i}.jpg")
        return str(img_dir)

    def test_predict_single_output_format(self, sample_image):
        """Test that predict_single returns correct format."""
        from src.model.classifier import SatelliteClassifier
        from src.model.predict import predict_single

        model = SatelliteClassifier(pretrained=False)
        model.eval()
        device = torch.device("cpu")
        model.to(device)

        result = predict_single(model, sample_image, device)

        assert "class" in result, "Result must contain 'class' key"
        assert "confidence" in result, "Result must contain 'confidence' key"
        assert "probabilities" in result, "Result must contain 'probabilities' key"

        # Validate class is one of the known classes
        assert result["class"] in CLASS_NAMES, (
            f"Predicted class '{result['class']}' not in known classes"
        )

        # Validate confidence is between 0 and 1
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confidence {result['confidence']} not in [0, 1]"
        )

        # Validate probabilities cover all classes
        assert len(result["probabilities"]) == NUM_CLASSES

    def test_predict_batch_output(self, sample_image_dir):
        """Test that predict_batch returns a list of predictions."""
        from src.model.classifier import SatelliteClassifier
        from src.model.predict import predict_batch

        model = SatelliteClassifier(pretrained=False)
        model.eval()
        device = torch.device("cpu")
        model.to(device)

        results = predict_batch(model, sample_image_dir, device)

        assert isinstance(results, list), "Batch results should be a list"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"

        for result in results:
            assert "class" in result
            assert "confidence" in result


class TestDataset:
    """Tests for the dataset loading utilities."""

    @pytest.fixture
    def mock_eurosat_dir(self, tmp_path):
        """Create a mock EuroSAT directory structure with sample images."""
        for class_name in CLASS_NAMES[:3]:  # Use only 3 classes for speed
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(20):  # 20 images per class
                img = Image.fromarray(
                    np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                )
                img.save(class_dir / f"{class_name}_{i:04d}.jpg")
        return str(tmp_path)

    def test_get_data_loaders(self, mock_eurosat_dir):
        """Test that data loaders are created with correct splits."""
        from src.model.dataset import get_data_loaders

        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=mock_eurosat_dir,
            batch_size=8,
            train_ratio=0.7,
            val_ratio=0.15,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check that train set is the largest
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        total = train_size + val_size + test_size

        assert total == 60, f"Expected 60 total images, got {total}"
        assert train_size > val_size, "Train set should be larger than val set"
        assert train_size > test_size, "Train set should be larger than test set"

    def test_data_loader_batch_shape(self, mock_eurosat_dir):
        """Test that batches have the expected tensor shapes."""
        from src.model.dataset import get_data_loaders

        train_loader, _, _ = get_data_loaders(
            data_dir=mock_eurosat_dir,
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
        )

        images, labels = next(iter(train_loader))

        assert images.ndim == 4, "Images should be 4D: (B, C, H, W)"
        assert images.shape[1] == 3, "Images should have 3 channels"
        assert labels.ndim == 1, "Labels should be 1D"
