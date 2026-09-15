# EuroSAT Dataset for Land Use and Land Cover (LULC) Classification

This directory contains the EuroSAT RGB dataset download scripts and documentation for the Satellite Imagery Analyzer project.

---

## 1. Overview & Dataset Description

The **EuroSAT** dataset is a benchmark dataset for Earth observation and remote sensing research, specifically designed for land use and land cover (LULC) classification based on Sentinel-2 satellite images.

### Key Specifications

| Property | Specification |
|---|---|
| **Satellite Source** | Sentinel-2 (European Space Agency / Copernicus Programme) |
| **Spectral Format** | RGB (3 color channels: Red, Green, Blue) |
| **Image Resolution** | $64 \times 64$ pixels |
| **Spatial Resolution** | 10 meters / pixel (each image spans $640\,\text{m} \times 640\,\text{m}$) |
| **File Format** | JPEG (8-bit per channel) |
| **Total Images** | 27,000 georeferenced satellite patches |
| **Number of Classes** | 10 distinct land-cover / land-use classes |
| **Geographic Coverage** | 34 European countries across multiple seasons |
| **Download Size** | ~90 MB compressed (`.zip`), ~130 MB uncompressed |

---

## 2. Dataset Classes

EuroSAT includes 10 land-cover and land-use categories with 2,000 to 3,000 images per class:

| # | Class Name | Description | Approximate Images |
|---|---|---|---|
| 0 | `AnnualCrop` | Agricultural fields planted with annual crops (e.g., wheat, corn) | 3,000 |
| 1 | `Forest` | Deciduous, evergreen, and mixed forest canopies | 3,000 |
| 2 | `HerbaceousVegetation` | Natural grasslands, meadows, and pastures | 3,000 |
| 3 | `Highway` | Highways, motorways, and multi-lane expressways | 2,500 |
| 4 | `Industrial` | Industrial zones, commercial complexes, and logistics hubs | 2,500 |
| 5 | `Pasture` | Fenced grazing land and pastures for livestock | 2,000 |
| 6 | `PermanentCrop` | Long-term agricultural crops (e.g., vineyards, fruit orchards) | 2,500 |
| 7 | `Residential` | Urban housing, suburban developments, and residential structures | 3,000 |
| 8 | `River` | Flowing surface water bodies, streams, rivers, and canals | 2,500 |
| 9 | `SeaLake` | Deep open water, inland lakes, coastal water, and sea zones | 3,000 |
| | **Total** | | **27,000** |

---

## 3. Download Instructions

### Method A: Automated Download via Python Script (Recommended)

The dataset download script `download_eurosat.py` automates downloading the `.zip` archive from Zenodo, verifies file integrity, extracts the 10 class directories, checks disk space, displays progress with `tqdm`, and outputs a class distribution summary.

```bash
# From the project root:
python data/download_eurosat.py
```

#### Command-Line Options:
```bash
# Specify a custom target directory (default: data/EuroSAT_RGB/)
python data/download_eurosat.py --output-dir /path/to/custom/dir

# Keep the downloaded zip file after extraction
python data/download_eurosat.py --keep-zip

# Force re-download even if already downloaded
python data/download_eurosat.py --force

# Verify an existing dataset installation without downloading
python data/download_eurosat.py --verify-only
```

---

### Method B: Manual Download

If automated download is unavailable or behind a proxy:

1. **Download the archive**:
   - **URL**: [https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip](https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip)
   - Using `curl`:
     ```bash
     curl -L -o data/EuroSAT_RGB.zip https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip
     ```
   - Using `wget`:
     ```bash
     wget -O data/EuroSAT_RGB.zip https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip
     ```

2. **Extract the archive**:
   ```bash
   unzip -q data/EuroSAT_RGB.zip -d data/
   ```

3. **Normalize directory structure**:
   The Zenodo archive packages the class folders under an internal directory named `2750/`. Move or rename this directory to `EuroSAT_RGB/`:
   ```bash
   mv data/2750 data/EuroSAT_RGB
   rm data/EuroSAT_RGB.zip
   ```

---

## 4. Expected Directory Structure

After downloading and extracting, the `data/` directory will look like this:

```
satellite-imagery-analyzer/
├── data/
│   ├── download_eurosat.py       # Download and verification script
│   ├── README.md                 # Dataset documentation (this file)
│   └── EuroSAT_RGB/              # Extracted dataset directory (EUROSAT_DIR)
│       ├── AnnualCrop/
│       │   ├── AnnualCrop_1.jpg
│       │   ├── AnnualCrop_2.jpg
│       │   └── ...
│       ├── Forest/
│       │   ├── Forest_1.jpg
│       │   └── ...
│       ├── HerbaceousVegetation/
│       ├── Highway/
│       ├── Industrial/
│       ├── Pasture/
│       ├── PermanentCrop/
│       ├── Residential/
│       ├── River/
│       └── SeaLake/
```

This layout is directly compatible with PyTorch's `torchvision.datasets.ImageFolder` and maps to `src.config.EUROSAT_DIR`.

---

## 5. File Sizes & System Requirements

- **Compressed Archive**: ~90 MB (`EuroSAT_RGB.zip`)
- **Extracted Directory**: ~130 MB (27,000 JPEG images)
- **Minimum Free Disk Space**: At least 300 MB free space recommended for the download, extraction, and staging process.

---

## 6. Citation and Credits

If you use this dataset in research or academic work, please credit the original authors:

> Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).  
> **EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.**  
> *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226.  
> DOI: [10.1109/JSTARS.2019.2918242](https://doi.org/10.1109/JSTARS.2019.2918242)

Zenodo Open-Access Archive:
- DOI: [10.5281/zenodo.7711810](https://doi.org/10.5281/zenodo.7711810)
- License: Open Access / Creative Commons Attribution
