# Data Quality Analyzer (DQA) - Demo
# Developer:  Brock Frary
# Date:  2025-09-26
#
# -------- Bootstrap Environment:Install Libraries, Import Utils, Load Dataset --------
# Imports
import os, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from datetime import datetime, timezone
from jinja2 import Template
from io import StringIO
import importlib
import warnings

# ---- Clear the terminal screen ----
def clear_screen():
    """Clear the console screen for a fresh run (cross-platform)."""
    os.system('cls' if os.name == 'nt' else 'clear')

# Define project paths
clear_screen()   # <-- clear screen here before printing path info

# Install and load libraries
print("\n============================================================\n")
print("\nData Quality Analyzer - Demo Vesion\n")
print("\n============================================================")
("\n============================================================\n")
print("\n- Make sure you've installed the requirements.txt from the root (python -m pip install -r requirements.txt)")
print("- Upload your .CSV file to analyze to the DATA_PATH (/data)")
print("     - Ctrl + C to keyboard interrupt the script, upload file, rerun script\n")

print("============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|    Section 1 – Environment Setup                   |")
print("|                                                    |")
print("+----------------------------------------------------+\n")

# Define project paths
print("Update PROJECT_PATH with full path to the root of DQA\n")
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.getenv("DQA_PROJECT_PATH", script_dir)

if not os.path.exists(PROJECT_PATH):
    print("\n[WARNING] PROJECT_PATH not found. Resetting to current script directory...")
    print(f"  Old PROJECT_PATH: {PROJECT_PATH}")
    PROJECT_PATH = script_dir
    print(f"  New PROJECT_PATH: {PROJECT_PATH}")
else:
    print("Checking PROJECT_PATH:", PROJECT_PATH, "-> Exists?", os.path.exists(PROJECT_PATH))

# Always rebuild derived paths based on final PROJECT_PATH
SRC_PATH  = os.path.abspath(os.path.join(PROJECT_PATH, "src"))
DST_PATH  = os.path.abspath(os.path.join(PROJECT_PATH, "dst"))
DATA_PATH = os.path.abspath(os.path.join(PROJECT_PATH, "data"))

print("Checking SRC_PATH:", SRC_PATH, "-> Exists?", os.path.exists(SRC_PATH))
print("Checking DST_PATH:", DST_PATH, "-> Exists?", os.path.exists(DST_PATH))
print("Checking DATA_PATH:", DATA_PATH, "-> Exists?", os.path.exists(DATA_PATH))

# Ensure SRC_PATH is in sys.path for custom module imports
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print("Added SRC_PATH to sys.path:", SRC_PATH)

# 3. Try to import custom modules
try:
    import dqa_utils as dqa
    importlib.reload(dqa)
    print("\nSuccessfully imported dqa_utils as dqa")
except Exception as e:
    print("\nCould not import dqa_utils:", e)

# Try to import file_picker
try:
    from file_picker import pick_file
    importlib.reload(sys.modules['file_picker'])
    print("Successfully imported file_picker and pick_file() is available.")
except Exception as e:
    print("Could not import file_picker:", e)

print("\n============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|    Section 2 – Choose Data File                    |")
print("|                                                    |")
print("+----------------------------------------------------+\n")
# Select dataset and load into pandas
# Only proceed if file_picker was successfully imported
if 'pick_file' in locals():
    try:
        file_path = pick_file(base_dir=DATA_PATH)

        if file_path is None:
            raise FileNotFoundError(
                "No dataset selected. Please upload a .csv file into "
                f"'{DATA_PATH}' and re-run."
            )

        print(f"[FILE PICKER] Dataset selected: {file_path}")
        df = pd.read_csv(file_path)
        print("\nDataset loaded successfully.\n")

    except FileNotFoundError as e:
        print("[FILE PICKER] ERROR:", e)
        df = None
    except Exception as e:
        print(f"[FILE PICKER] An unexpected error occurred while loading the dataset: {e}")
        df = None
else:
    print("\nSkipping dataset selection and loading due to file_picker import failure.")
    df = None

# -------- Display Data Quality Summary --------
print("============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|    Section 3 – Data Quality Summary                |")
print("|                                                    |")
print("+----------------------------------------------------+\n")
def summarize_dataframe(df: pd.DataFrame, n_rows: int = 5):
    # ---------------- Print basic shape ----------------
    print("="*60)
    print("\nDataFrame Shape (rows, columns):", df.shape)

    # ---------------- Show column information ----------------
    print("\nColumn Information:")
    print("-"*60)
    df.info()

    # ---------------- Convert numeric-like columns to float ----------------
    # This ensures that columns containing non-numeric characters (e.g., "M", "--", or spaces)
    # will be coerced into proper floats, with invalid entries converted to NaN.

    # Convert any object columns that are actually numeric
    for col in df.select_dtypes(include="object").columns:
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            # If conversion produced at least some numeric values, keep it
            if converted.notna().sum() > 0:
                df[col] = converted
        except Exception:
            # Leave column unchanged if conversion isn't meaningful
            pass

    # Try to parse any object columns that look like datetimes
    for col in df.select_dtypes(include="object").columns:
        try:
            converted = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
            # Only keep conversion if at least half the rows were valid datetimes
            if converted.notna().sum() > (0.5 * len(df)):
                df[col] = converted
        except Exception:
            pass

    # ---------------- Display first n rows ----------------
    print("\nFirst 5 rows (head):")
    print("-"*60)
    print(df.head(n_rows))

    # ---------------- Display last n rows ----------------
    print("\nLast 5 rows (tail):")
    print("-"*60)
    print(df.tail(n_rows))

    # ---------------- Descriptive statistics (numeric only) ----------------
    print("\nDescriptive Statistics (numeric columns):")
    print("-"*60)
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        print(numeric_df.describe())
    else:
        print("No numeric columns found.")

    # ---------------- End of summary ----------------
    print("="*60)

# -------- Call it explicitly --------
if df is None:
    print("[ERROR] No dataset loaded. Aborting.")
    sys.exit(1)
summarize_dataframe(df)

# --------  Determine missing values in the data --------
print("\n+----------------------------------------------------+")
print("|                                                    |")
print("|    Section 4 – Missing Values Per Column           |")
print("|                                                    |")
print("+----------------------------------------------------+\n")
print("Missing values per column:\n")

# Calculate missing counts and percentages
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

# Combine into a DataFrame for clarity
missing_df = pd.DataFrame({
    "Missing Count": missing_counts,
    "Missing %": missing_pct.round(2)
}).sort_values(by="Missing Count", ascending=False)

print((missing_df.to_string()))

# ----------- Export: Self-Contained HTML Report -----------
# Destination folder: use DST_PATH defined in Bootstrap cell
dst_folder = DST_PATH
os.makedirs(dst_folder, exist_ok=True)

# Extract original dataset filename from file_path (selected via file_picker)
input_file = file_path if 'file_path' in globals() and file_path else "unknown_dataset.csv"
input_filename = os.path.basename(input_file)

# Create timestamp in UTC (Python 3.12+ safe, timezone-aware)
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")

# Build output filename: timestamp + input filename + .html
output_filename = f"{timestamp}-{input_filename}.html"

# Full report path
report_path = os.path.join(dst_folder, output_filename)

# Report generation message
print("============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|           Section 5 – Report Generation            |")
print("|                                                    |")
print("+----------------------------------------------------+\n")
print("[REPORT] Report generation in progress...  (this may take a few minutes depending on dataset size)")

# Attempt export with error handling
try:
    dqa.export_html_report(df, output_path=report_path)
    print(f"[REPORT] Report generation successful.")
    print(f"[REPORT] Saved to: {report_path}")
    print("")

except Exception as e:
    print(f"[ERROR] Failed to export HTML report.\nReason: {e}")
    print("\n--- Troubleshooting ---")
    print("1. Check if 'df' is defined and contains a valid DataFrame.")
    print("2. Ensure 'dqa_utils.py' is imported as 'dqa'.")
    print("3. Verify that the destination folder exists.")
    print(f"   Destination path checked: {dst_folder}")
    print("4. If the error persists, re-run the bootstrap cell to re-import modules.")

    # © 2025 Brock Frary. All rights reserved.