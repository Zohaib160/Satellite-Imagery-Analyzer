"""
Download and extract the EuroSAT RGB dataset.

This script downloads the EuroSAT RGB dataset from Zenodo, extracts it
to the target directory (default: data/EuroSAT_RGB/), verifies the presence
of all 10 land-use / land-cover classes, and displays a summary of image counts.

Dataset reference:
    Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
    EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and
    Land Cover Classification. IEEE J-STARS, 12(7), 2217-2226.
    Zenodo DOI: 10.5281/zenodo.7711810
"""

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

# Attempt importing tqdm with a fallback for environments without it
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        """Minimal fallback progress indicator when tqdm is not installed."""

        def __init__(
            self,
            iterable=None,
            total: Optional[int] = None,
            unit: str = "it",
            unit_scale: bool = False,
            unit_divisor: int = 1000,
            desc: str = "",
            **kwargs,
        ):
            self.iterable = iterable
            self.total = total
            self.unit = unit
            self.unit_scale = unit_scale
            self.unit_divisor = unit_divisor
            self.desc = desc
            self.n = 0
            self._iter = iter(iterable) if iterable is not None else None

        def __iter__(self):
            return self

        def __next__(self):
            if self._iter is None:
                raise StopIteration
            try:
                item = next(self._iter)
                self.update(1)
                return item
            except StopIteration:
                sys.stderr.write("\n")
                sys.stderr.flush()
                raise

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.write("\n")
            sys.stderr.flush()

        def update(self, n: int = 1):
            self.n += n
            if self.total and self.total > 0:
                pct = (self.n / self.total) * 100
                if self.unit_scale:
                    scale = self.unit_divisor * self.unit_divisor
                    n_s = self.n / scale
                    tot_s = self.total / scale
                    msg = f"\r{self.desc}: {pct:5.1f}% [{n_s:.1f}/{tot_s:.1f} M{self.unit}]"
                else:
                    msg = f"\r{self.desc}: {pct:5.1f}% [{self.n}/{self.total} {self.unit}]"
            else:
                if self.unit_scale:
                    scale = self.unit_divisor * self.unit_divisor
                    n_s = self.n / scale
                    msg = f"\r{self.desc}: {n_s:.1f} M{self.unit}"
                else:
                    msg = f"\r{self.desc}: {self.n} {self.unit}"
            sys.stderr.write(msg)
            sys.stderr.flush()


# Zenodo direct download URL for EuroSAT RGB
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"

# Standard EuroSAT 10 land-use / land-cover classes
DEFAULT_CLASS_NAMES = [
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

# Supported image file extensions
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Required disk space in bytes (~250 MB for zip download + extraction)
MIN_REQUIRED_DISK_BYTES = 250 * 1024 * 1024

# Resolve project root and attempt importing config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import EUROSAT_DIR, CLASS_NAMES
except ImportError:
    EUROSAT_DIR = PROJECT_ROOT / "data" / "EuroSAT_RGB"
    CLASS_NAMES = DEFAULT_CLASS_NAMES


def check_disk_space(target_path: Path, required_bytes: int = MIN_REQUIRED_DISK_BYTES) -> None:
    """
    Check if the target storage volume has sufficient free space.

    Args:
        target_path: Path where data will be stored.
        required_bytes: Number of free bytes required.

    Raises:
        OSError: If available disk space is less than required.
    """
    check_dir = target_path
    while not check_dir.exists() and check_dir != check_dir.parent:
        check_dir = check_dir.parent

    try:
        usage = shutil.disk_usage(check_dir)
    except OSError as e:
        sys.stderr.write(f"Warning: Could not check disk usage: {e}\n")
        return

    if usage.free < required_bytes:
        free_mb = usage.free / (1024 * 1024)
        req_mb = required_bytes / (1024 * 1024)
        raise OSError(
            f"Insufficient disk space on volume for '{target_path}'. "
            f"Available: {free_mb:.1f} MB, Required: {req_mb:.1f} MB."
        )


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 64) -> Path:
    """
    Download a file from a URL to the specified destination with a progress bar.

    Args:
        url: Remote file URL to download.
        dest_path: Local filesystem path where the file will be saved.
        chunk_size: Stream buffer size in bytes (default: 64 KB).

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: On download errors (HTTP, network, timeout, or I/O).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download_path = dest_path.with_suffix(dest_path.suffix + ".partial")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "SatelliteImageryAnalyzer/1.0 (EuroSAT Downloader)"
        )
    }

    req = urllib.request.Request(url, headers=headers)
    print(f"Connecting to: {url}")

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            content_length_header = response.headers.get("Content-Length")
            total_size = int(content_length_header) if content_length_header else None

            if total_size:
                size_mb = total_size / (1024 * 1024)
                print(f"Download size: {size_mb:.2f} MB")

            with open(temp_download_path, "wb") as out_file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {dest_path.name}",
                ) as pbar:
                    while True:
                        try:
                            chunk = response.read(chunk_size)
                        except (OSError, TimeoutError) as read_err:
                            raise RuntimeError(f"Network error while reading stream: {read_err}") from read_err

                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(len(chunk))

        # Atomic rename once complete
        if temp_download_path.exists():
            temp_download_path.replace(dest_path)

        print(f"Download completed successfully: {dest_path}")
        return dest_path

    except urllib.error.HTTPError as e:
        if temp_download_path.exists():
            temp_download_path.unlink()
        raise RuntimeError(f"HTTP error {e.code} while downloading {url}: {e.reason}") from e

    except urllib.error.URLError as e:
        if temp_download_path.exists():
            temp_download_path.unlink()
        raise RuntimeError(f"Network error connecting to {url}: {e.reason}") from e

    except (OSError, TimeoutError) as e:
        if temp_download_path.exists():
            temp_download_path.unlink()
        raise RuntimeError(f"I/O or timeout error during download: {e}") from e

    except Exception:
        if temp_download_path.exists():
            temp_download_path.unlink()
        raise


def extract_dataset(zip_path: Path, output_dir: Path) -> None:
    """
    Extract the EuroSAT zip archive to the output directory.

    Handles root folder normalization if the archive wraps classes inside
    a subfolder (such as '2750/' or 'EuroSAT_RGB/').

    Args:
        zip_path: Path to the .zip archive.
        output_dir: Destination directory for the extracted dataset.

    Raises:
        zipfile.BadZipFile: If the archive is corrupt or incomplete.
        RuntimeError: If extraction fails.
    """
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"'{zip_path}' is not a valid zip archive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting '{zip_path.name}' to '{output_dir}'...")

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = archive.infolist()
            with tqdm(members, desc="Extracting files", unit="file") as pbar:
                for member in pbar:
                    archive.extract(member, path=output_dir)

    except (zipfile.BadZipFile, OSError) as e:
        raise RuntimeError(f"Failed to extract zip archive '{zip_path}': {e}") from e

    # Reorganize if extracted files are nested under a single wrapper directory
    # EuroSAT_RGB.zip typically extracts to an internal folder named '2750'
    candidate_wrapper_dirs = ["2750", "EuroSAT_RGB"]
    for wrapper_name in candidate_wrapper_dirs:
        wrapper_path = output_dir / wrapper_name
        if wrapper_path.is_dir() and wrapper_path.resolve() != output_dir.resolve():
            print(f"Found nested folder '{wrapper_name}', flattening into '{output_dir}'...")
            for item in list(wrapper_path.iterdir()):
                target = output_dir / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            try:
                wrapper_path.rmdir()
            except OSError:
                pass


def verify_dataset(dataset_dir: Path, expected_classes: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Verify that all expected class directories exist and contain valid images.

    Args:
        dataset_dir: Root directory of the extracted EuroSAT dataset.
        expected_classes: List of expected class directory names.

    Returns:
        Dictionary mapping class names to the count of images in each class.

    Raises:
        FileNotFoundError: If the dataset directory or any class directory is missing.
        RuntimeError: If any class directory has zero images.
    """
    if expected_classes is None:
        expected_classes = CLASS_NAMES

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"EuroSAT dataset directory not found: '{dataset_dir}'")

    missing_classes = []
    class_counts: Dict[str, int] = {}

    for class_name in expected_classes:
        class_folder = dataset_dir / class_name
        if not class_folder.is_dir():
            missing_classes.append(class_name)
        else:
            image_files = [
                f for f in class_folder.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ]
            class_counts[class_name] = len(image_files)

    if missing_classes:
        raise FileNotFoundError(
            f"EuroSAT dataset verification failed in '{dataset_dir}'.\n"
            f"Missing {len(missing_classes)} of {len(expected_classes)} class directories:\n"
            + "\n".join(f"  - {c}" for c in missing_classes)
        )

    empty_classes = [c for c, count in class_counts.items() if count == 0]
    if empty_classes:
        raise RuntimeError(
            f"Dataset verification failed: the following classes contain no images:\n"
            + "\n".join(f"  - {c}" for c in empty_classes)
        )

    return class_counts


def print_summary(class_counts: Dict[str, int], dataset_dir: Path) -> None:
    """
    Print a formatted summary table showing class counts and total images.

    Args:
        class_counts: Dictionary of {class_name: image_count}.
        dataset_dir: Path to the dataset directory.
    """
    total_images = sum(class_counts.values())
    divider = "=" * 60
    sub_divider = "-" * 60

    print("\n" + divider)
    print("EuroSAT RGB Dataset Summary")
    print(divider)
    print(f"Location:      {dataset_dir.resolve()}")
    print(f"Total Classes: {len(class_counts)}")
    print(sub_divider)
    print(f"{'Class Name':<35} {'Image Count':>15}")
    print(sub_divider)

    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name:<35} {count:>15,d}")

    print(sub_divider)
    print(f"{'Total Images':<35} {total_images:>15,d}")
    print(divider)
    print("Dataset verification: PASSED (All classes present and populated)\n")


def download_and_setup(
    output_dir: Path,
    url: str = EUROSAT_URL,
    keep_zip: bool = False,
    force: bool = False,
) -> Dict[str, int]:
    """
    Orchestrate disk checking, downloading, extraction, verification, and summary.

    Args:
        output_dir: Destination path for EuroSAT RGB dataset.
        url: Remote download URL for the archive.
        keep_zip: Whether to retain the downloaded zip file after extraction.
        force: If True, re-downloads and re-extracts even if already present.

    Returns:
        Dictionary of class image counts.
    """
    # Check if dataset is already present and valid
    if not force and output_dir.is_dir():
        try:
            counts = verify_dataset(output_dir)
            print(f"Dataset already exists and is verified at '{output_dir}'.")
            print("Use --force to re-download and overwrite.")
            print_summary(counts, output_dir)
            return counts
        except (FileNotFoundError, RuntimeError):
            print(f"Existing directory '{output_dir}' is incomplete or invalid. Proceeding with download...")

    # Check available disk space
    check_disk_space(output_dir)

    # Temporary zip archive location
    zip_filename = Path(url.split("?")[0]).name or "EuroSAT_RGB.zip"
    zip_path = output_dir.parent / zip_filename

    # Download archive
    download_file(url, zip_path)

    # Extract archive
    try:
        extract_dataset(zip_path, output_dir)
    finally:
        if not keep_zip and zip_path.exists():
            print(f"Cleaning up archive file '{zip_path.name}'...")
            zip_path.unlink()

    # Verify and summarize
    counts = verify_dataset(output_dir)
    print_summary(counts, output_dir)
    return counts


def main() -> None:
    """Command-line entry point for downloading EuroSAT dataset."""
    parser = argparse.ArgumentParser(
        description="Download, extract, verify, and summarize the EuroSAT RGB dataset."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=EUROSAT_DIR,
        help=f"Directory to extract EuroSAT dataset into (default: {EUROSAT_DIR})",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=EUROSAT_URL,
        help=f"Download URL for the EuroSAT RGB zip file (default: {EUROSAT_URL})",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded .zip archive after extraction",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download and re-extraction even if the dataset already exists",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing dataset directory without downloading",
    )

    args = parser.parse_args()

    try:
        if args.verify_only:
            counts = verify_dataset(args.output_dir)
            print_summary(counts, args.output_dir)
        else:
            download_and_setup(
                output_dir=args.output_dir,
                url=args.url,
                keep_zip=args.keep_zip,
                force=args.force,
            )
    except KeyboardInterrupt:
        sys.stderr.write("\nOperation cancelled by user.\n")
        sys.exit(130)
    except Exception as exc:
        sys.stderr.write(f"\nError: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
