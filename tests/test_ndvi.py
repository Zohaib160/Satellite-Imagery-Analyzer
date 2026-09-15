"""
Tests for the NDVI analysis and change detection modules.

Tests NDVI calculation, vegetation classification, change detection,
and edge case handling.
"""

import pytest
import numpy as np

from src.config import NDVI_THRESHOLDS, CHANGE_THRESHOLD


class TestNDVICalculation:
    """Tests for NDVI computation."""

    def test_basic_ndvi(self):
        """Test NDVI calculation with known values."""
        from src.analysis.ndvi import calculate_ndvi

        # NIR=0.5, Red=0.1 → NDVI = (0.5-0.1)/(0.5+0.1) = 0.4/0.6 ≈ 0.6667
        nir = np.array([[0.5]], dtype=np.float32)
        red = np.array([[0.1]], dtype=np.float32)
        ndvi = calculate_ndvi(nir, red)

        expected = (0.5 - 0.1) / (0.5 + 0.1)
        np.testing.assert_almost_equal(ndvi[0, 0], expected, decimal=4)

    def test_ndvi_range(self):
        """Test that NDVI values stay within [-1, 1]."""
        from src.analysis.ndvi import calculate_ndvi

        np.random.seed(42)
        nir = np.random.rand(100, 100).astype(np.float32)
        red = np.random.rand(100, 100).astype(np.float32)
        ndvi = calculate_ndvi(nir, red)

        assert ndvi.min() >= -1.0, f"NDVI min {ndvi.min()} is below -1"
        assert ndvi.max() <= 1.0, f"NDVI max {ndvi.max()} is above 1"

    def test_ndvi_division_by_zero(self):
        """Test that NDVI handles zero denominator (NIR + Red = 0)."""
        from src.analysis.ndvi import calculate_ndvi

        nir = np.array([[0.0, 0.5]], dtype=np.float32)
        red = np.array([[0.0, 0.3]], dtype=np.float32)
        ndvi = calculate_ndvi(nir, red)

        # When both are 0, NDVI should be 0 (not NaN or Inf)
        assert not np.isnan(ndvi).any(), "NDVI contains NaN values"
        assert not np.isinf(ndvi).any(), "NDVI contains Inf values"
        assert ndvi[0, 0] == 0.0, "NDVI should be 0 when NIR and Red are both 0"

    def test_ndvi_pure_vegetation(self):
        """Test high NDVI for vegetation-like reflectance (high NIR, low Red)."""
        from src.analysis.ndvi import calculate_ndvi

        nir = np.array([[0.8]], dtype=np.float32)
        red = np.array([[0.1]], dtype=np.float32)
        ndvi = calculate_ndvi(nir, red)

        assert ndvi[0, 0] > 0.5, "Dense vegetation should have NDVI > 0.5"

    def test_ndvi_water(self):
        """Test negative NDVI for water-like reflectance (low NIR, higher Red)."""
        from src.analysis.ndvi import calculate_ndvi

        nir = np.array([[0.05]], dtype=np.float32)
        red = np.array([[0.3]], dtype=np.float32)
        ndvi = calculate_ndvi(nir, red)

        assert ndvi[0, 0] < 0.0, "Water should have negative NDVI"

    def test_ndvi_array_shapes(self):
        """Test that output shape matches input shape."""
        from src.analysis.ndvi import calculate_ndvi

        for shape in [(10, 10), (64, 64), (256, 256)]:
            nir = np.random.rand(*shape).astype(np.float32)
            red = np.random.rand(*shape).astype(np.float32)
            ndvi = calculate_ndvi(nir, red)
            assert ndvi.shape == shape, f"Expected shape {shape}, got {ndvi.shape}"


class TestVegetationClassification:
    """Tests for vegetation classification from NDVI values."""

    def test_classify_water(self):
        """Test that negative NDVI is classified as Water."""
        from src.analysis.ndvi import classify_vegetation

        ndvi = np.array([[-0.5, -0.2]], dtype=np.float32)
        classes = classify_vegetation(ndvi)

        assert classes[0, 0] == "Water"
        assert classes[0, 1] == "Water"

    def test_classify_dense_vegetation(self):
        """Test that high NDVI is classified as Dense Vegetation."""
        from src.analysis.ndvi import classify_vegetation

        ndvi = np.array([[0.8, 0.7]], dtype=np.float32)
        classes = classify_vegetation(ndvi)

        assert classes[0, 0] == "Dense Vegetation"
        assert classes[0, 1] == "Dense Vegetation"

    def test_classify_all_categories(self):
        """Test that all NDVI ranges map to expected categories."""
        from src.analysis.ndvi import classify_vegetation

        # One value per category
        ndvi = np.array([[-0.5, 0.05, 0.2, 0.45, 0.8]], dtype=np.float32)
        classes = classify_vegetation(ndvi)

        assert classes[0, 0] == "Water"
        assert classes[0, 1] == "Barren/Built-up"
        assert classes[0, 2] == "Sparse Vegetation"
        assert classes[0, 3] == "Moderate Vegetation"
        assert classes[0, 4] == "Dense Vegetation"


class TestNDVIStatistics:
    """Tests for NDVI statistics computation."""

    def test_compute_statistics(self):
        """Test that statistics dict contains expected keys."""
        from src.analysis.ndvi import compute_statistics

        ndvi = np.random.uniform(-0.2, 0.8, (100, 100)).astype(np.float32)
        stats = compute_statistics(ndvi)

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_statistics_values(self):
        """Test statistics with known array."""
        from src.analysis.ndvi import compute_statistics

        ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        stats = compute_statistics(ndvi)

        np.testing.assert_almost_equal(stats["min"], 0.2, decimal=4)
        np.testing.assert_almost_equal(stats["max"], 0.8, decimal=4)
        np.testing.assert_almost_equal(stats["mean"], 0.5, decimal=4)


class TestSampleDataGeneration:
    """Tests for synthetic sample data generation."""

    def test_generate_sample_data(self):
        """Test that sample data generator produces valid arrays."""
        from src.analysis.ndvi import generate_sample_data

        nir, red = generate_sample_data(height=128, width=128)

        assert nir.shape == (128, 128), f"NIR shape {nir.shape} != (128, 128)"
        assert red.shape == (128, 128), f"Red shape {red.shape} != (128, 128)"
        assert nir.min() >= 0.0, "NIR values should be non-negative"
        assert red.min() >= 0.0, "Red values should be non-negative"


class TestChangeDetection:
    """Tests for temporal change detection."""

    def test_detect_changes_no_change(self):
        """Test with identical NDVI arrays (no change)."""
        from src.analysis.change_detection import detect_changes

        ndvi = np.full((50, 50), 0.5, dtype=np.float32)
        result = detect_changes(ndvi, ndvi)

        assert "difference_map" in result
        assert "change_mask" in result
        assert "statistics" in result
        assert result["statistics"]["percent_changed"] == 0.0

    def test_detect_changes_with_loss(self):
        """Test detection of vegetation loss."""
        from src.analysis.change_detection import detect_changes

        before = np.full((50, 50), 0.7, dtype=np.float32)  # Dense vegetation
        after = np.full((50, 50), 0.2, dtype=np.float32)    # Sparse vegetation

        result = detect_changes(before, after, threshold=0.1)

        assert result["statistics"]["loss_pixels"] > 0
        assert result["statistics"]["percent_loss"] > 0
        assert result["difference_map"].min() < 0  # Negative = loss

    def test_detect_changes_with_gain(self):
        """Test detection of vegetation gain."""
        from src.analysis.change_detection import detect_changes

        before = np.full((50, 50), 0.2, dtype=np.float32)
        after = np.full((50, 50), 0.7, dtype=np.float32)

        result = detect_changes(before, after, threshold=0.1)

        assert result["statistics"]["gain_pixels"] > 0
        assert result["statistics"]["percent_gain"] > 0

    def test_change_mask_values(self):
        """Test that change mask only contains -1, 0, 1."""
        from src.analysis.change_detection import detect_changes

        np.random.seed(42)
        before = np.random.uniform(0.0, 1.0, (100, 100)).astype(np.float32)
        after = np.random.uniform(0.0, 1.0, (100, 100)).astype(np.float32)

        result = detect_changes(before, after)
        unique_values = set(np.unique(result["change_mask"]))

        assert unique_values.issubset({-1, 0, 1}), (
            f"Change mask contains unexpected values: {unique_values}"
        )

    def test_generate_change_report(self):
        """Test that change report is a non-empty string."""
        from src.analysis.change_detection import detect_changes, generate_change_report

        before = np.full((50, 50), 0.7, dtype=np.float32)
        after = np.full((50, 50), 0.3, dtype=np.float32)
        result = detect_changes(before, after)

        report = generate_change_report(result)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_generate_sample_change_data(self):
        """Test sample change data generation."""
        from src.analysis.change_detection import generate_sample_change_data

        before, after = generate_sample_change_data(height=128, width=128)

        assert before.shape == (128, 128)
        assert after.shape == (128, 128)
        # The arrays should differ (simulated change)
        assert not np.array_equal(before, after), "Before and after should differ"
