"""
Unit tests for data/download_eurosat.py.
"""

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.download_eurosat import (
    CLASS_NAMES,
    check_disk_space,
    extract_dataset,
    print_summary,
    verify_dataset,
)


class TestDownloadEuroSAT(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_disk_space_success(self):
        """Test disk space check succeeds when required bytes is small."""
        # 1 KB required should succeed on any machine
        check_disk_space(self.base_path, required_bytes=1024)

    def test_check_disk_space_insufficient(self):
        """Test disk space check raises OSError when required bytes is impossibly huge."""
        with self.assertRaises(OSError) as ctx:
            check_disk_space(self.base_path, required_bytes=10**18)
        self.assertIn("Insufficient disk space", str(ctx.exception))

    def test_verify_dataset_missing_dir(self):
        """Test verify_dataset raises FileNotFoundError if root directory doesn't exist."""
        non_existent = self.base_path / "does_not_exist"
        with self.assertRaises(FileNotFoundError):
            verify_dataset(non_existent)

    def test_verify_dataset_missing_classes(self):
        """Test verify_dataset raises FileNotFoundError if class directories are missing."""
        dataset_dir = self.base_path / "incomplete_dataset"
        dataset_dir.mkdir()
        # Only create 1 class directory instead of 10
        (dataset_dir / CLASS_NAMES[0]).mkdir()
        (dataset_dir / CLASS_NAMES[0] / "test.jpg").write_text("dummy")

        with self.assertRaises(FileNotFoundError) as ctx:
            verify_dataset(dataset_dir)
        self.assertIn("Missing", str(ctx.exception))

    def test_verify_dataset_empty_class(self):
        """Test verify_dataset raises RuntimeError if a class directory is empty."""
        dataset_dir = self.base_path / "empty_class_dataset"
        dataset_dir.mkdir()
        for class_name in CLASS_NAMES:
            (dataset_dir / class_name).mkdir()

        # Classes are empty, should fail
        with self.assertRaises(RuntimeError) as ctx:
            verify_dataset(dataset_dir)
        self.assertIn("contain no images", str(ctx.exception))

    def test_verify_dataset_success(self):
        """Test verify_dataset returns counts when all classes are properly populated."""
        dataset_dir = self.base_path / "valid_dataset"
        dataset_dir.mkdir()
        for idx, class_name in enumerate(CLASS_NAMES, start=1):
            class_dir = dataset_dir / class_name
            class_dir.mkdir()
            for img_idx in range(idx):
                (class_dir / f"img_{img_idx}.jpg").write_text("dummy image data")

        counts = verify_dataset(dataset_dir)
        self.assertEqual(len(counts), 10)
        for idx, class_name in enumerate(CLASS_NAMES, start=1):
            self.assertEqual(counts[class_name], idx)

    def test_extract_dataset_flattens_2750(self):
        """Test extract_dataset correctly flattens '2750/' wrapper folder from Zenodo archive."""
        zip_path = self.base_path / "mock_archive.zip"
        output_dir = self.base_path / "extracted_data"

        # Create a mock zip with 2750/ wrapper
        with zipfile.ZipFile(zip_path, "w") as zf:
            for class_name in CLASS_NAMES:
                zf.writestr(f"2750/{class_name}/sample.jpg", b"image content")

        extract_dataset(zip_path, output_dir)

        # Ensure wrapper folder is gone and class folders are directly under output_dir
        self.assertFalse((output_dir / "2750").exists())
        counts = verify_dataset(output_dir)
        self.assertEqual(len(counts), 10)
        for class_name in CLASS_NAMES:
            self.assertEqual(counts[class_name], 1)

    def test_print_summary(self):
        """Test print_summary prints without raising errors."""
        mock_counts = {c: 100 for c in CLASS_NAMES}
        # Call print_summary and ensure it doesn't fail
        print_summary(mock_counts, self.base_path)


if __name__ == "__main__":
    unittest.main()
