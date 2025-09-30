# Data Quality Analyzer (DQA) v2.7.2

**Video Demo**: Go to the `DQA-v2.6 folder` -> click on the `DQA_demo_v2.6.1_2025-09-27.mp4`, then click `View Raw` to download the video to run locally.

## Overview
The **Data Quality Analyzer (DQA)** is a Python-based application designed to analyze datasets (CSV) for **missing values, descriptive statistics, and overall data quality**.  
It produces a **self-contained HTML report** with tables and charts that can be downloaded and reviewed.  

Two versions are provided:
- **Live** (`DQA_live-v2.7.2.py`) – allows you to select a dataset (.csv) at runtime.
- **Demo** (`DQA_demo-v2.7.2.py`) – includes a static sample dataset, used to demonstrate features without requiring uploads.

> Note: `.ipynb` notebooks were supported in v2.5 for Colab.  
> They are **not updated in v2.7.2**. The most current version of DQA is the local `.py` scripts.

---

## Project Structure
```
DQA/        # Root project folder
│
├── DAQ_demo_v2.6.1_2025-09-27.mp4	# Downloadable video demonstration
│
├── live/                     # Live version (user-selectable CSVs)
│   ├── DQA_live-v2.6.py      # Local execution script (Python 3.12+)
│   ├── data/                 # Input datasets (.csv placed here)
│   ├── docs/                 # Best practices documents
│   ├── dst/                  # Output folder for generated HTML reports (ignored in GitHub)
│   └── src/                  # Helper modules
│       ├── dqa_utils.py      # Functions for charting + HTML export
│       └── file_picker.py    # Environment-aware dataset selector
│
├── demo/                     # Demo version (self-contained, static dataset)
│   ├── DQA_demo.py
│   ├── data/                 # Input datasets (.csv here)
│   │   └── sample.csv        # Static dataset bundled with demo
│   ├── docs/                 # Best practices documents
│   ├── dst/                  # Output folder for generated HTML reports (ignored in GitHub)
│   └── src/                  # Helper modules
│       ├── dqa_utils.py      # Functions for charting + HTML export
│       └── file_picker.py    # Environment-aware dataset selector
```

> **Ignored via `.gitignore`:**  
> - `__pycache__/`, `*.pyc`, `*.pyo`  
> - `/dst/*.html` (generated reports)  
> - `/dst/plots/` (generated plots)  
> - `.env`  

---

## Features
- **Descriptive Statistics** (numeric + categorical)
- **Missing Values Report** (counts, percentages, severity coloring)
- **Duplicate Rows** (counts + severity indicator)
- **Outlier Detection** (Z-score method, severity coloring)
- **Correlation Heatmap** (with severity thresholds)
- **Export as HTML** (self-contained, embeddable, shareable)
- **Environment-Aware File Picker** (`file_picker.py`) for both Colab and Windows
- **Dynamic Path Handling** (`PROJECT_PATH`, `DATA_PATH`, `DST_PATH`)

---

## Installation (Local / Windows 11 + PowerShell / Python 3.12+)

1. Clone the repo:
   ```powershell
   git clone https://github.com/<your-repo>/data-quality-analyzer.git
   cd data-quality-analyzer/live
   ```

2. Create a Python virtual environment (for a clean local install, recommended, but not required):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts -activate
   ```

3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Install wkhtmltopdf (HTML -> PDF for reporting, download and install - may require additional configuration)

5. Run the script:
   ```powershell
   python DQA_live-v2.7.2.py
   ```

---

## Usage

- Place input `.csv` files into:
  - `data-quality-analyzer/live/data/`
  - or for demo: `data-quality-analyzer/demo/data/`

- Reports are exported into:
  - `data-quality-analyzer/live/dst/`
  - or for demo: `data-quality-analyzer/demo/dst/`

- Open the `.html` report in any browser.  
  *(Reports and plots are excluded from GitHub but generated locally.)*

---

## Example Output

The generated report includes:
- Dataset shape, column info, head/tail views
- Missing values table with severity coloring
- Duplicate row analysis
- Descriptive statistics (numeric + categorical)
- Outlier detection
- Correlation analysis with heatmap

*Footer example:*
```
DQA - Brock Frary - v2.6.1 - 2025-09-28
```

---

## Requirements
See `requirements.txt` for dependencies:
- pandas  
- numpy  
- matplotlib  
- seaborn  
- jinja2
- PyYAML

---

## Developer
**Brock Frary**  
Version: v2.7.2 (as of 2025-09-30)  

---

© 2025 Brock Frary. All rights reserved.