"""
Tests for the FastAPI web server endpoints.

Uses httpx.AsyncClient to test API endpoints without starting
a live server.
"""

import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Generate a sample satellite image as bytes."""
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_grayscale_bytes():
    """Generate a sample grayscale band image as bytes."""
    img_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="L")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test that health endpoint returns 200 with correct payload."""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_check_method(self, client):
        """Test that health endpoint only accepts GET."""
        response = client.post("/api/health")
        assert response.status_code == 405


class TestClassesEndpoint:
    """Tests for the classes listing endpoint."""

    def test_get_classes(self, client):
        """Test that classes endpoint returns all 10 EuroSAT classes."""
        response = client.get("/api/classes")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list) or "classes" in data

    def test_classes_count(self, client):
        """Test that we get exactly 10 classes."""
        response = client.get("/api/classes")
        data = response.json()

        classes = data if isinstance(data, list) else data.get("classes", [])
        assert len(classes) == 10, f"Expected 10 classes, got {len(classes)}"


class TestClassifyEndpoint:
    """Tests for the image classification endpoint."""

    def test_classify_without_file(self, client):
        """Test that classify endpoint returns error without file."""
        response = client.post("/api/classify")
        assert response.status_code == 422  # Validation error

    def test_classify_with_image(self, client, sample_image_bytes):
        """Test classification with a sample image."""
        response = client.post(
            "/api/classify",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        # Either 200 (model loaded) or 503 (model not available)
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = response.json()
            assert "class" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert isinstance(data["probabilities"], dict)


class TestNDVIEndpoint:
    """Tests for the NDVI analysis endpoint."""

    def test_ndvi_demo_mode(self, client):
        """Test NDVI endpoint with demo data."""
        response = client.post("/api/ndvi?demo=true")

        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data

    def test_ndvi_with_files(self, client, sample_grayscale_bytes):
        """Test NDVI with uploaded band images."""
        response = client.post(
            "/api/ndvi",
            files={
                "nir": ("nir.png", sample_grayscale_bytes, "image/png"),
                "red": ("red.png", sample_grayscale_bytes, "image/png"),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data


class TestChangeDetectEndpoint:
    """Tests for the change detection endpoint."""

    def test_change_detect_demo_mode(self, client):
        """Test change detection with demo data."""
        response = client.post("/api/change-detect?demo=true")

        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data

    def test_change_detect_with_files(self, client, sample_grayscale_bytes):
        """Test change detection with uploaded images."""
        response = client.post(
            "/api/change-detect",
            files={
                "before": ("before.png", sample_grayscale_bytes, "image/png"),
                "after": ("after.png", sample_grayscale_bytes, "image/png"),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_endpoint(self, client):
        """Test that invalid endpoints return 404."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_classify_invalid_file(self, client):
        """Test classification with invalid file type."""
        response = client.post(
            "/api/classify",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        # Should return an error (400 or 422)
        assert response.status_code in (400, 422, 500)
