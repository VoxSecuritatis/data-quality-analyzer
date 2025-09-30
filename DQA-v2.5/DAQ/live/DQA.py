# Data Quality Analyzer (DQA)
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
print("\nData Quality Analyzer - Live Vesion\n")
print("\n============================================================")
("\n============================================================\n")
print("\nMake sure you've installed the requirements.txt from the root (python -m pip install -r requirements.txt)\n")

print("============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|    Section 1 – Environment Setup                   |")
print("|                                                    |")
print("+----------------------------------------------------+\n")

# Define project paths
print("Update PROJECT_PATH with full path to the root of DQA\n")
PROJECT_PATH = r"D:\Python_Projects\Data_Quality_Analyzer\DQA\live" # Project path
SRC_PATH     = os.path.abspath(os.path.join(PROJECT_PATH, "src"))   # Source path (/src)
DST_PATH     = os.path.abspath(os.path.join(PROJECT_PATH, "dst"))   # Destination path (/dst)
DATA_PATH    = os.path.abspath(os.path.join(PROJECT_PATH, "data"))  # Data path (/data)

# Add src/ to sys.path (normalize, no duplicates) BEFORE importing custom modules
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Display paths
print("Checking PROJECT_PATH:", PROJECT_PATH, "-> Exists?", os.path.exists(PROJECT_PATH))
print("Checking SRC_PATH:", SRC_PATH, "-> Exists?", os.path.exists(SRC_PATH))
print("Checking DST_PATH:", DST_PATH, "-> Exists?", os.path.exists(DST_PATH))
print("Checking DATA_PATH:", DATA_PATH, "-> Exists?", os.path.exists(DATA_PATH))

# Check if critical directories exist
if not os.path.exists(SRC_PATH):
    print(f"\n[ERROR] Source directory not found: {SRC_PATH}")
    print("Please ensure you have the 'src' folder within your DQA/live directory in Google Drive.")
elif not os.path.exists(DATA_PATH):
    print(f"\n[ERROR] Data directory not found: {DATA_PATH}")
    print("Please ensure you have the 'data' folder within your DQA/live directory in Google Drive.")
else:
    print("\nProject directories look good.")

print("\nsys.path includes:")
print("SRC_PATH:", SRC_PATH)
print("DST_PATH:", DST_PATH)
print("DATA_PATH:", DATA_PATH)
print("PROJECT_PATH:", PROJECT_PATH)

# Try to import custom modules AFTER adding src to sys.path
print("\n--- Sanity check for helper files ---")
dqa_utils_path = os.path.join(SRC_PATH, "dqa_utils.py")
file_picker_path = os.path.join(SRC_PATH, "file_picker.py")

print("Looking for dqa_utils.py at:", dqa_utils_path)
print("Exists?", os.path.exists(dqa_utils_path))

print("Looking for file_picker.py at:", file_picker_path)
print("Exists?", os.path.exists(file_picker_path))

# 5. Try to import custom modules
try:
    import dqa_utils as dqa
    importlib.reload(dqa)
    print("\nSuccessfully imported dqa_utils as dqa")
except Exception as e:
    print("\nCould not import dqa_utils:", e)

try:
    from file_picker import pick_file
    print("Successfully imported pick_file from file_picker")
except Exception as e:
    print("Could not import file_picker:", e)
print("")

print("============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|    Section 2 – Choose Data File                    |")
print("|                                                    |")
print("+----------------------------------------------------+\n")
# Select dataset and load into pandas
# Only proceed if file_picker was successfully imported
if 'pick_file' in locals():
    try:
        # Check if DATA_PATH exists before calling pick_file
        if not os.path.exists(DATA_PATH):
             raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")

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
            converted = pd.to_datetime(df[col], errors="coerce")
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

# ----------------- Charting Section (Data Quality, Univariant, Multivariant)

print("\n============================================================\n")
print("+----------------------------------------------------+")
print("|                                                    |")
print("|              Section 5 – Charting                  |")
print("|                                                    |")
print("+----------------------------------------------------+\n")
print("[CHARTING] Starting chart generation...\n")

# Helper: group plots into rows of up to 3 charts
def plot_in_grid(plot_funcs, titles, section_name):
    """Display plots in rows of 3 with shared formatting."""
    n = len(plot_funcs)
    if n == 0:
        print(f"\n[CHARTING] No charts to generate for {section_name}.")
        return

    print(f"\n[CHARTING] Generating {n} plots for {section_name} (max 3 per row).")

    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i, func in enumerate(plot_funcs):
        plt.sca(axes[i])
        func()
        axes[i].set_title(titles[i], fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.show()

# ----------- 1) Data Quality Statistics (Single Full-Width Plots) -----------
print("+---------------------------------------------+")
print("|                                             |")
print("|  Subsection 1 – Data Quality Statistics     |")
print("|                                             |")
print("+---------------------------------------------+")

# Missing values per column
missing_pct = df.isnull().mean() * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
if not missing_pct.empty:
    sns.barplot(
        x=missing_pct.values,
        y=missing_pct.index,
        color="red",
    )
    plt.xlabel("Percentage Missing")
    plt.ylabel("Columns")
    plt.title("Missing Values per Column (%)")
    plt.show()
else:
    print("\n[CHARTING] No missing values detected.\n")

# Duplicate counts
dup_counts = pd.Series({
    "Unique": len(df) - df.duplicated().sum(),
    "Duplicates": df.duplicated().sum()
})
sns.barplot(
    x=dup_counts.index,
    y=dup_counts.values,
    color="red",
)
plt.title("Unique vs Duplicate Rows")
plt.show()

# Data type distribution
dtype_counts = df.dtypes.value_counts()
sns.barplot(
    x=dtype_counts.index.astype(str),
    y=dtype_counts.values,
    color="red",
)
plt.title("Data Types Distribution")
plt.ylabel("Count")
plt.show()

# ----------- 2) Univariate Analysis (Grouped 3 per Row) -----------
print("+---------------------------------------------+")
print("|                                             |")
print("|  Subsection 2 – Univariant Analysis         |")
print("|                                             |")
print("+---------------------------------------------+")

# Numeric distributions (limit to 10)
num_cols = df.select_dtypes(include=np.number).columns
if len(num_cols) > 10:
    print(f"\n[WARNING] Dataset has {len(num_cols)} numeric columns. Showing first 10 only.\n")
    skipped = list(num_cols[10:])
    print("Skipped numeric columns:", skipped)
    choice = input("Render ALL numeric distributions instead? (y/N): ").strip().lower()
    if choice == "y":
        num_cols = df.select_dtypes(include=np.number).columns
    else:
        num_cols = num_cols[:10]

plot_funcs = [lambda col=col: sns.histplot(df[col].dropna(), kde=True, color="red") for col in num_cols]
titles = [f"Distribution: {col}" for col in num_cols]
plot_in_grid(plot_funcs, titles, "Numeric Distributions")

# Numeric boxplots (limit to 10)
num_cols = df.select_dtypes(include=np.number).columns
if len(num_cols) > 10:
    print(f"\n[WARNING] Dataset has {len(num_cols)} numeric columns. Showing first 10 only.\n")
    skipped = list(num_cols[10:])
    print("Skipped numeric columns:", skipped)
    choice = input("Render ALL numeric boxplots instead? (y/N): ").strip().lower()
    if choice == "y":
        num_cols = df.select_dtypes(include=np.number).columns
    else:
        num_cols = num_cols[:10]

plot_funcs = [lambda col=col: sns.boxplot(x=df[col], color="red") for col in num_cols]
titles = [f"Boxplot: {col}" for col in num_cols]
plot_in_grid(plot_funcs, titles, "Numeric Boxplots")

# Categorical frequencies (limit to 10)
cat_cols = df.select_dtypes(include="object").columns
if len(cat_cols) > 10:
    print(f"\n[WARNING] Dataset has {len(cat_cols)} categorical columns. Showing first 10 only.\n")
    skipped = list(cat_cols[10:])
    print("Skipped categorical columns:", skipped)
    choice = input("Render ALL categorical frequencies instead? (y/N): ").strip().lower()
    if choice == "y":
        cat_cols = df.select_dtypes(include="object").columns
    else:
        cat_cols = cat_cols[:10]

plot_funcs = [
    lambda col=col: sns.countplot(
        data=df,
        y=col,
        order=df[col].value_counts().index,
        color="red",
    )
    for col in cat_cols
]
titles = [f"Categorical Frequency: {col}" for col in cat_cols]
plot_in_grid(plot_funcs, titles, "Categorical Frequencies")

# ----------- 3) Multivariate Analysis (Optional, On Request) -----------
print("\n+---------------------------------------------+")
print("|                                             |")
print("|  Subsection 3 – Multivariant Analysis       |")
print("|                (Optional)                   |")
print("|                                             |")
print("+---------------------------------------------+\n")

run_multi = input("Run multivariate analysis? (y/N)\n(Note: this process takes up to 5 minutes to finish): ").strip().lower()

if run_multi == "y":
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10,8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="Reds")
        plt.title("Correlation Heatmap (Numeric Variables)")
        plt.show()
    else:
        print("\n[CHARTING] Not enough numeric columns for correlation heatmap.\n")

    # Pairplot
    if len(numeric_cols) > 1:
        sns.pairplot(df[numeric_cols].dropna(), diag_kind="kde", plot_kws={"color": "red"})
        plt.suptitle("Pairplot of Numeric Variables", y=1.02)
        plt.show()

    # Scatterplots between numeric variables
    if len(numeric_cols) >= 2:
        plot_funcs = [
            (lambda i=i: sns.scatterplot(x=df[numeric_cols[i]], y=df[numeric_cols[i+1]], color="red"))
            for i in range(len(numeric_cols)-1)
        ]
        titles = [f"Scatterplot: {numeric_cols[i]} vs {numeric_cols[i+1]}" for i in range(len(numeric_cols)-1)]
        plot_in_grid(plot_funcs, titles, "Scatterplots (Numeric Pairs)")

    # --- Numeric vs Categorical Boxplots (capped to avoid overload) ---
        max_plots = 8  # adjust this number as you like
        plot_pairs = [(num, cat) for num in numeric_cols for cat in cat_cols][:max_plots]

        sample_size = min(5000, len(df)) # Dynamic sample size: use all rows if small dataset, else cap at 5000
        plot_funcs = [
            (lambda num=num, cat=cat: sns.boxplot(
                x=df[cat].sample(n=min(sample_size, len(df)), random_state=42),
                y=df[num].sample(n=min(sample_size, len(df)), random_state=42),
                color="red"
            ))
            for num, cat in plot_pairs
        ]
        titles = [f"Boxplot: {num} by {cat}" for num, cat in plot_pairs]

        plot_in_grid(plot_funcs, titles, "Numeric vs Categorical Boxplots")

        # Countplot comparisons between categorical variables
        if len(cat_cols) >= 2:
            max_pairs = 5  # cap to avoid overload
            cat_pairs = [(c1, c2) for i, c1 in enumerate(cat_cols) for c2 in cat_cols[i+1:]]
            plot_funcs = [
                (lambda c1=c1, c2=c2: sns.countplot(x=df[c1], hue=df[c2], palette="Reds"))
                for c1, c2 in cat_pairs[:max_pairs]
            ]
            titles = [f"Countplot: {c1} vs {c2}" for c1, c2 in cat_pairs[:max_pairs]]
            plot_in_grid(plot_funcs, titles, "Categorical vs Categorical Countplot")
        else:
            print("\n[CHARTING] NOTE: Not enough categorical columns for categorical vs categorical plots. This is not an error, just an observation.")

print("\n[CHARTING] Chart generation complete.\n")

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
print("|           Section 6 – Report Generation            |")
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