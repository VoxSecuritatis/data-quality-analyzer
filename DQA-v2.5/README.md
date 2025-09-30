# Data Quality Analyzer (DQA)

## Overview
The **Data Quality Analyzer (DQA)** is a Python-based application designed to analyze datasets (CSV) for **missing values, descriptive statistics, and overall data quality**.  
It produces a **self-contained HTML report** with tables and charts that can be downloaded and reviewed.  

Two versions are provided:
- **Live** (`DQA.py` / `DQA.ipynb`): allows you to select a dataset (.csv) at runtime.
- **Demo** (`DQA_demo.py` / `DQA_demo.ipynb`): includes a static sample dataset, used to demonstrate features without requiring file uploads.

Local or Google Colab provided:
- **.py** (`DQA.py` or `DQA_demo.py`):  local Python scripts
- **.ipynb** (`DQA.ipynb` or `DQA_demo.ipynb`):  Google Colab Jupyter Notebooks, intended to be used on Google Drive

---

## Project Structure
```
DQA/                 # Root project folder
│
├── live/            # Live version (user-selectable CSVs)
│   ├── DQA.py       # Local execution script (Python 3.12+)
│   ├── DQA.ipynb    # Colab notebook for running in Google Drive/Colab
│   ├── data/        # Input datasets (.csv placed here)
│   ├── dst/         # Output folder for generated HTML reports
│   └── src/         # Helper modules
│       ├── dqa_utils.py   # Functions for charting + HTML export
│       └── file_picker.py # Colab-friendly file selector
│
├── demo/            # Demo version (self-contained, static dataset)
│   ├── DQA_demo.py
│   ├── DQA_demo.ipynb
│   ├── data/        # Input datasets (.csv here)
│   |   └── sample.csv   # Static dataset bundled with demo
│   ├── dst/         # Output folder for generated HTML reports
│   └── src/         # Helper modules
│       ├── dqa_utils.py   # Functions for charting + HTML export
│       └── file_picker.py # Colab-friendly file selector│   
```

---

## Features
- **Descriptive Statistics** (numeric + categorical)
- **Missing Values Report** (counts, percentages, bar chart)
- **Correlation Heatmap** (sampled for performance)
- **Export as HTML** (self-contained, embeddable, shareable)
- **Colab File Picker** (`file_picker.py`) for selecting datasets inside Google Drive  
- **Sampling for Performance** – default: 1000 rows, configurable

---

## Installation (Local / Windows 11 + PowerShell + VSCode)
1. Clone or copy the repo into a folder:
   ```powershell
   git clone https://github.com/<your-repo>/DQA.git
   cd DQA/live
   ```

2. Create a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Run the script:
   ```powershell
   python DQA.py
   ```

---

## Installation (Google Colab)
1. Upload the **entire `DQA/` folder** into the root of **My Drive**.
   ```
   My Drive/
     └── DQA/
         └── live/
		     ├── data/ 
	    	 ├── dst/ 
    		 └── src/
         └── demo/
            ├── data/ 
	    	 ├── dst/ 
    		 └── src/
2. Open **`DQA.ipynb`** (for live) or **`DQA_demo.ipynb`** (for demo).  
3. Mount Google Drive when prompted.  
4. Run all cells → a file picker will allow you to choose from uploaded `.csv` files.  
5. For the first run, you may need to update your libraries - you will be prompted to restart the session.  Restart session and run all.
---

## Usage
- Place input `.csv` files into:
  - **Local**: `DQA/live/data/`
  - **Colab**: `/My Drive/DQA/live/data/`

- Reports are exported into:
  - **Local**: `DQA/live/dst/`
  - **Colab**: `/My Drive/DQA/live/dst/`

- Open the `.html` report in any browser.

---

## Example Output
The generated report includes:
- Dataset shape, column info, head/tail views
- Missing values table + bar chart
- Descriptive statistics (numeric + categorical)
- Correlation heatmap (if ≥2 numeric columns)
- Footer: `DQA - Brock Frary - v2.5 - 2025-09-25`

---

## Requirements
See `requirements.txt` for dependencies:
- pandas  
- numpy  
- matplotlib  
- seaborn  
- jinja2  

---

## Developer
**Brock Frary**  
Version: v2.5 (as of 2025-09-25)  

---

© 2025 Brock Frary. All rights reserved.