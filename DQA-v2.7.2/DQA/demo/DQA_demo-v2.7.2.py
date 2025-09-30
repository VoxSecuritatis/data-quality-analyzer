# Data Quality Analyzer (DQA) - Live
# Version 2.7.0
# Developer:  Brock Frary
# Date:  2025-09-28
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
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pathlib
from pathlib import Path

# ---- Clear the terminal screen ----
def clear_screen():
    """Clear the console screen for a fresh run (cross-platform)."""
    os.system('cls' if os.name == 'nt' else 'clear')

# Define project paths
clear_screen()   # <-- clear screen here before printing path info

# Install and load libraries
print("\n============================================================\n")
print("\nData Quality Analyzer - Live Vesion\n")
print("\n============================================================\n")
print("\n- Make sure you've installed the requirements.txt from the root (python -m pip install -r requirements.txt)")
print("- Upload your .CSV file to analyze to the DATA_PATH (/data)")
print("     - Exit (Ctrl + C) the script, upload file, rerun script\n")

print("============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|    Section 1 – Environment Setup                   |")
print("|                                                    |")
print("+----------------------------------------------------+\n")

# ============================================================
# Define project paths (dynamic: live/ or demo/ aware)
# ============================================================
print("Update PROJECT_PATH with full path to the root of DQA\n")

from pathlib import Path

# Current script directory (…\DQA-vX.Y.Z\DQA\live  OR  …\DQA-vX.Y.Z\DQA\demo)
script_dir = Path(__file__).resolve().parent

# Project root is the parent of the run-context folder (…\DQA-vX.Y.Z\DQA)
project_root_default = script_dir.parent
PROJECT_PATH = os.getenv("DQA_PROJECT_PATH", str(project_root_default))

if not os.path.exists(PROJECT_PATH):
    print("\n[WARNING] PROJECT_PATH not found. Resetting to script_dir.parent…")
    print(f"  Old PROJECT_PATH: {PROJECT_PATH}")
    PROJECT_PATH = str(project_root_default)
    print(f"  New PROJECT_PATH: {PROJECT_PATH}")
else:
    print("Checking PROJECT_PATH:", PROJECT_PATH, "-> Exists?", os.path.exists(PROJECT_PATH))

# Prefer run-context (live/ or demo/) subfolders; fallback to project-root subfolders
run_ctx = script_dir  # live/ or demo/

# --- SRC_PATH ---
src_live_demo = run_ctx / "src"
src_project   = Path(PROJECT_PATH) / "src"
SRC_PATH = str(src_live_demo if src_live_demo.exists() else src_project)

# --- DST_PATH (create if missing) ---
dst_live_demo = run_ctx / "dst"
dst_project   = Path(PROJECT_PATH) / "dst"
DST_PATH = str(dst_live_demo if dst_live_demo.exists() else dst_project)
if not os.path.exists(DST_PATH):
    try:
        os.makedirs(DST_PATH, exist_ok=True)
        print("[PATH] Created DST_PATH:", DST_PATH)
    except Exception as e:
        print("[WARNING] Could not create DST_PATH:", DST_PATH, "Reason:", e)

# --- DATA_PATH (no auto-create; just pick best available) ---
data_live_demo = run_ctx / "data"
data_project   = Path(PROJECT_PATH) / "data"
DATA_PATH = str(data_live_demo if data_live_demo.exists() else data_project)

print("Checking SRC_PATH:", SRC_PATH, "-> Exists?", os.path.exists(SRC_PATH))
print("Checking DST_PATH:", DST_PATH, "-> Exists?", os.path.exists(DST_PATH))
print("Checking DATA_PATH:", DATA_PATH, "-> Exists?", os.path.exists(DATA_PATH))

# Ensure SRC_PATH is in sys.path for custom module imports
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    print("Added SRC_PATH to sys.path:", SRC_PATH)

# 3) Try to import custom modules
try:
    import dqa_utils as dqa
    importlib.reload(dqa)
    print("\nSuccessfully imported dqa_utils as dqa")
except Exception as e:
    print("\nCould not import dqa_utils:", e)
    # Hard stop if dqa import fails; avoids 'name dqa is not defined' later
    raise

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
print("+----------------------------------------------------+")
# Select dataset and load into pandas
# Only proceed if file_picker was successfully imported
if 'pick_file' in locals():
    try:
        file_path = pick_file(base_dir=DATA_PATH)

        if file_path is None:
            print("\n[FILE PICKER] No file selected.")
            print(f"\n[INFO] Exiting — no dataset selected. CSV source datasets must be placed in: {DATA_PATH}\n")
            sys.exit(0)  # clean exit before proceeding further

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

# ----------- Imputation Recommendations -----------
print("\n+----------------------------------------------------+")
print("|                                                    |")
print("|    Section 5 – Imputation Recommendations          |")
print("|                                                    |")
print("+----------------------------------------------------+\n")

# Define df_cleaned before the try/except so it always exists
df_cleaned = df.copy()

try:
    # Build recommendations directly
    recs = []
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64]:
                strategy = "Mean / Median"
            elif df[col].dtype == "object":
                strategy = "Most Frequent"
            else:
                strategy = "Custom (domain-specific)"
            recs.append({
                "Column": col,
                "Missing %": round(df[col].isnull().mean() * 100, 2),
                "Suggestions": strategy
            })

    recs_df = pd.DataFrame(recs)

    impute_html = "<h2>Imputation Recommendations</h2>"
    impute_html += """
    <ul>
        <li>These are suggested strategies to handle missing values.</li>
        <li>Numeric columns → Mean or Median imputation.</li>
        <li>Categorical columns → Most Frequent (mode).</li>
        <li>Other datatypes may need domain-specific handling.</li>
    </ul>
    """ + recs_df.to_html(index=False, classes="table table-striped", escape=False)

    print("[IMPUTE] Imputation recommendations generated successfully.")
except Exception as e:
    print(f"[IMPUTE] Failed to generate/apply imputation recommendations.\nReason: {e}")
    impute_html = f"<h2>Imputation Recommendations</h2><p>Error: {e}</p>"

# ----------- Export: Self-Contained HTML Report -----------
# Destination folder: use DST_PATH defined in Bootstrap cell
dst_folder = DST_PATH
os.makedirs(dst_folder, exist_ok=True)

# Extract original dataset filename from file_path (selected via file_picker)
input_file = file_path if 'file_path' in globals() and file_path else "unknown_dataset.csv"
input_filename = os.path.basename(input_file)

# Create timestamp in UTC (Python 3.12+ safe, timezone-aware)
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")

# Build output filename: timestamp + dataset base name + .html
base_name = os.path.splitext(input_filename)[0]  # strip .csv or other extension
output_filename = f"{timestamp}-{base_name}.html"

# Full report path
report_path = os.path.join(dst_folder, output_filename)

# Report generation message
print("============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|           Section 6 – Report Generation            |")
print("|                                                    |")
print("+----------------------------------------------------+\n")
compare_mode_input = input("Enable comparison mode (original vs. cleaned)? (y/N): ").strip().lower()
comparison_mode = (compare_mode_input == "y")
if comparison_mode:
    print("\n[WARNING] Comparison mode enabled — report generation may take longer on large datasets.")
else:
    print("\n[INFO] Comparison mode disabled — report will be based on original dataset only.")

print("\n[REPORT] Report generation in progress...")

# Attempt export with error handling
try:
    # FIX: always generate standardized TIMESTAMP-<base>.html report
    dqa.export_html_report(df, output_path=report_path, comparison_mode=comparison_mode)
    print(f"[REPORT] Report generation successful.")
    print(f"[REPORT] HTML report saved to: {report_path}")
    print("")
    
    # ----------- Export: Cleaned Dataset CSV -----------
    cleaned_filename = f"CLEANED_{timestamp}-{input_filename}"
    cleaned_path = os.path.join(dst_folder, cleaned_filename)

    try:
        # Ensure a cleaned dataset exists
        if 'df_cleaned' in globals():
            export_df = df_cleaned
        elif 'cleaned_df' in globals():
            export_df = cleaned_df
        else:
            print("[WARNING] No cleaned dataset found — exporting original dataset instead.")
            export_df = df

        export_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

        print(f"[CLEANED] Cleaned dataset export successful.")
        print(f"[CLEANED] Saved to: {cleaned_path}")
    except Exception as e:
        print(f"[ERROR] Failed to export cleaned dataset.\nReason: {e}")

    # ----------- Final Recap Section -----------
    print("\n========================= SUMMARY ==========================")
    print("\n[SUMMARY] Run completed successfully.\n")

    # Original dataset file path (unaltered dataset)
    original_path = os.path.join(dst_folder, input_filename)

    # Copy original dataset to dst folder
    import shutil
    try:
        shutil.copy(input_file, original_path)
        print(f"[ORIGINAL] Original dataset copied to: {original_path}")
    except Exception as e:
        print(f"[ERROR] Failed to copy original dataset to /dst.\nReason: {e}")

    # Compute schema paths from the report path (suffix-safe)
    base_html = Path(report_path)
    original_schema_path = str(base_html.with_suffix("")) + "-original.schema.expected.yaml"
    cleaned_schema_path = str(base_html.with_suffix("")) + "-cleaned.schema.expected.yaml"

    print(f"\n[REPORT]  HTML report saved to: {report_path}")
    print(f"[CLEANED] Cleaned dataset saved to: {cleaned_path}")

    # Always show Original Schema path
    print(f"[SCHEMA]  Original Schema saved to: {original_schema_path}")

    # Cleaned Schema path (only generated in comparison mode)
    if comparison_mode:
        print(f"[SCHEMA]  Cleaned Schema saved to: {cleaned_schema_path}")
    else:
        print("[SCHEMA]  Cleaned dataset schema not generated — run with comparison mode enabled.")
    
    print("\n============================================================")

except Exception as e:
    print(f"[ERROR] Failed to export HTML report.\nReason: {e}")
    print("\n--- Troubleshooting ---")
    print("1. Check if 'df' is defined and contains a valid DataFrame.")
    print("2. Ensure 'dqa_utils.py' is imported as 'dqa'.")
    print("3. Verify that the destination folder exists.")
