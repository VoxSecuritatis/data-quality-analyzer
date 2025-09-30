# dqa_utils.py
# Version: 2.7.2
# Developer: Brock Frary
# Date: 2025-09-30

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from jinja2 import Template
import base64
from io import StringIO
import re
import numpy as np
import yaml
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from bs4 import BeautifulSoup

# --- NLTK resource checks (cached once) ---
try:
    nltk.data.find("tokenizers/punkt")
    _HAVE_PUNKT = True
except LookupError:
    print("[WARN] NLTK 'punkt' tokenizer not found. Run once:\n"
          ">>> import nltk\n>>> nltk.download('punkt')")
    _HAVE_PUNKT = False

try:
    _STOPWORDS = set(stopwords.words("english"))
    _HAVE_STOPWORDS = True
except LookupError:
    print("[WARN] NLTK 'stopwords' corpus not found. Run once:\n"
          ">>> import nltk\n>>> nltk.download('stopwords')")
    _STOPWORDS = set()
    _HAVE_STOPWORDS = False


try:
    thresholds_path = Path(__file__).resolve().parent.parent / "docs" / "thresholds.csv"
    thresholds_df = pd.read_csv(thresholds_path)
except Exception as e:
    thresholds_df = pd.DataFrame()
    print(f"[WARNING] Could not load thresholds.csv: {e}")

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# ----------- Helper: Encode Image as Base64 -----------
def encode_image_base64(path: str) -> str:
    """
    Encode an image file as a base64 string for embedding in HTML.

    Parameters
    ----------
    path : Path
        Path object pointing to the image file.

    Returns
    -------
    str
        A base64-encoded string representation of the image.

    Notes
    -----
    - Used to embed plots (PNG files) directly into the HTML report.
    - Ensures portability of the generated report.
    - If the file cannot be read, returns an empty string.
    """
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# ----------- Helper: Sanitize Filenames -----------
def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use as a filename.

    Parameters
    ----------
    name : str
        The original filename string.

    Returns
    -------
    str
        A safe filename string with problematic characters replaced
        by underscores.

    Notes
    -----
    - Ensures cross-platform compatibility when saving reports.
    - Preserves alphanumeric, dash, underscore, and dot characters.
    """
    return re.sub(r'[\\/*?:"<>| ]', "_", str(name))

# ----------- Helper: Threshold Assignment -----------
def get_cutoffs(check_name: str, default_green: float, default_yellow: float):
    """
    Robustly parse threshold text for a given check. Accepts either percent
    (e.g., '85% ... 90%') or proportion (e.g., '0.85 ... 0.90') formats.
    Falls back to provided defaults if parsing fails.
    """
    green_cutoff, yellow_cutoff = float(default_green), float(default_yellow)
    try:
        row = thresholds_df[thresholds_df["Check"].str.contains(check_name, case=False)].iloc[0]
        rule_text = str(row["Threshold / Rule of Thumb"])

        # Grab floats, not just integers (handles '0.85' and '85')
        nums = re.findall(r"\d+(?:\.\d+)?", rule_text)
        vals = [float(n) for n in nums]

        if len(vals) >= 2:
            # If all numbers look like proportions (<= 1), convert to percents
            if all(v <= 1.0 for v in vals):
                vals = [v * 100.0 for v in vals]

            # First number = green upper bound, yellow = max of the rest
            green_cutoff = float(vals[0])
            yellow_cutoff = float(max(vals[1:]))

    except Exception:
        # Keep defaults on any issue
        pass

    return green_cutoff, yellow_cutoff

# ----------- Helper: Colorize Severity -----------
def colorize_severity(label: str) -> str:
    """
    Return an HTML color-coded string based on a severity label.

    Parameters
    ----------
    label : str
        Severity label such as "Low", "Medium", or "High".

    Returns
    -------
    str
        A formatted HTML string that colors the label:
        - Green for "Low"
        - Yellow for "Medium"
        - Red for "High"
    """
    colors = {
        "Green":  "#28a745",
        "Yellow": "#ffc107",
        "Red":    "#dc3545"
    }
    color = colors.get(label, "gray")
    text_color = "black" if label == "Yellow" else "white"
    return f'<span style="color:{text_color}; background-color:{color}; padding:2px 6px; border-radius:4px;">{label}</span>'

# ----------- Helper: DataFrame Summary to HTML -----------
def summarize_dataframe_html(df: pd.DataFrame, n_rows: int = 5) -> str:
    try:
        print("[SUMMARY] Starting DataFrame summary to HTML...")

        html_parts = []

        # --- Shape ---
        html_parts.append("<h2>DataFrame Shape</h2>")
        html_parts.append(
            "<ul>"
            "<li>Indicates dataset size. Ensure enough rows for robust analysis, but not so many that compute becomes inefficient.</li>"
            "<li>Columns should balance completeness and usability—too few may omit important context, too many may introduce noise.</li>"
            "</ul>"
            "<h3>Sampling Best Practices</h3>"
            "<ul>"
                "<li><b>Datasets ≤ 100K rows</b>: profile the full dataset (no sampling needed).</li>"
                "<li><b>Datasets 100K–1M rows</b>: use a random sample of ~10K–50K rows.</li>"
                "<li><b>Datasets > 1M rows</b>: cap profiling at 50K–100K rows for efficiency.</li>"
                "<li><b>Always use random sampling</b> to avoid bias — first rows may not be representative.</li>"
                "<li><b>Stratified sampling</b> is recommended for categorical imbalance checks so rare classes aren’t lost.</li>"
                "<li><b>Why?</b> Past ~50K rows, key stats (mean, median, variance, correlations) stabilize with minimal gain from more rows.</li>"
            "</ul>"
        )
        html_parts.append(f"<p>{df.shape[0]} rows × {df.shape[1]} columns</p>")

        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        html_parts.append("<h2>Column Information</h2>")
        html_parts.append(
            "<ul>"
            "<li>Confirms column data types and memory usage.</li>"
            "<li>Numeric values should be stored as numeric types; categorical/text as category or string types.</li>"
            "<li>Incorrect types may slow analysis or yield misleading results.</li>"
            "</ul>"
        )
        html_parts.append(f"<pre>{info_str}</pre>")

        html_parts.append("<h2>First Rows (head)</h2>")
        html_parts.append(df.head(n_rows).to_html(classes="table table-striped", border=0))
        html_parts.append("<h2>Last Rows (tail)</h2>")
        html_parts.append(df.tail(n_rows).to_html(classes="table table-striped", border=0))

        try:
            html_parts.append("<h2>Descriptive Statistics (Numeric Columns)</h2>")
            html_parts.append(df.describe(include=[float, int]).to_html(classes="table table-striped", border=0))
        except Exception as e:
            html_parts.append(f"<p><i>No numeric columns found. ({e})</i></p>")

        try:
            html_parts.append("<h2>Descriptive Statistics (Categorical Columns)</h2>")
            html_parts.append(df.describe(include=[object]).to_html(classes="table table-striped", border=0))
        except Exception as e:
            html_parts.append(f"<p><i>No categorical columns found. ({e})</i></p>")

        print("[SUMMARY] DataFrame summary to HTML completed successfully.")
        return "\n".join(html_parts)

    except Exception as e:
        print(f"[SUMMARY] Error during DataFrame summary: {e}")
        return f"<h2>DataFrame Summary</h2><p>Error: {e}</p>"

# PDF export helper function
def export_pdf_report(output_path: str, sections: dict, title: str = "Data Quality Analyzer Report", images: dict = None) -> None:
    """
    Exports the data quality report to a PDF file using reportlab.

    Enhancements:
    - Preserves tables from HTML (renders as ReportLab tables).
    - Embeds charts/images (if provided in `images` dict).
    - Keeps explanatory text formatted.
    """
    try:
        print("[EXPORT] Starting improved PDF export...")

        # Setup PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 20))

        # Process each section
        for section, html_content in sections.items():
            story.append(Paragraph(f"<b>{section}</b>", styles['Heading2']))

            soup = BeautifulSoup(html_content, "html.parser")

            # Render tables
            for table in soup.find_all("table"):
                rows = []
                for tr in table.find_all("tr"):
                    row = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if row:
                        rows.append(row)
                if rows:
                    pdf_table = Table(rows, repeatRows=1)
                    pdf_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]))
                    story.append(pdf_table)
                    story.append(Spacer(1, 12))
                table.decompose()  # remove after processing

            # Render text paragraphs
            text_content = soup.get_text(separator="\n")
            for line in text_content.splitlines():
                if line.strip():
                    story.append(Paragraph(line.strip(), styles['Normal']))
            story.append(Spacer(1, 12))

            # Render charts/images if available
            if images and section in images:
                for img_path in images[section]:
                    try:
                        story.append(Image(img_path, width=400, height=300))
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        print(f"[EXPORT] Skipped image {img_path}: {e}")

            story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)
        print(f"[EXPORT] Improved PDF export completed: {output_path}")

    except Exception as e:
        print(f"[EXPORT] Error during improved PDF export: {e}")

# ----------- Export HTML Report -----------
def export_html_report(
    df: pd.DataFrame,
    output_path="dst/dqa_report.html",
    sample_size: int = 1000,
    comparison_mode: bool = False
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary_html = summarize_dataframe_html(df)
    plot_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    # ----------- Clean up plots folder -----------
    plots_dir = Path(output_path).parent / "plots"
    if plots_dir.exists() and plots_dir.is_dir():
        import shutil
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(exist_ok=True)

    # ----------- Missing Values -----------
    try:
        print("[MISSING] Starting missing value analysis...")

        missing_counts = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": missing_counts.values,
            "Missing %": missing_pct.round(2).values
        }).sort_values(by="Missing Count", ascending=False)

        g, y = get_cutoffs("Missing Values", 5, 30)

        def assign_missing_flag(pct: float) -> str:
            if pct < g:
                return "Green"
            elif pct <= y:
                return "Yellow"
            else:
                return "Red"

        missing_df["Severity"] = missing_df["Missing %"].apply(assign_missing_flag)
        missing_df["Severity"] = missing_df["Severity"].apply(colorize_severity)

        missing_html = """
        <h2>Missing Values per Column</h2>
        <ul>
            <li>Missing data can reduce the accuracy of models and analyses if not addressed properly.</li>
            <li>Patterns of missingness (random vs. systematic) often provide insights into data collection quality.</li>
            <li>Imputation, removal, or domain-specific handling are standard strategies for managing missing values.</li>
        </ul>
        <p><i>Best Practice: &lt;5% = Green, 5–30% = Yellow, &gt;30% = Red</i></p>
        """ + missing_df.to_html(classes="table table-striped", index=False, escape=False)

        print(f"[MISSING] Completed successfully: {len(missing_df)} columns analyzed.")
    except Exception as e:
        print(f"[MISSING] Error during missing value analysis: {e}")
        missing_html = f"<h2>Missing Values</h2><p>Error: {e}</p>"

    # ----------- Imputation Recommendations Section -----------
    try:
        print("[IMPUTATION] Generating imputation recommendations...")

        recs_df = get_imputation_recommendations(df)
        impute_html = "<h2>Imputation Recommendations</h2>"
        impute_html += """
        <ul>
            <li>Suggests how to handle missing values per column.</li>
            <li>Guides analysts in choosing appropriate replacements.</li>
            <li>Improves consistency and reproducibility of data preparation.</li>
        </ul>
        """
        impute_html += "<h3>Original Dataset</h3>"
        impute_html += """
        <ul>
            <li>Shows missing-value strategies directly from raw data.</li>
            <li>Helps analysts see baseline issues before cleaning.</li>
        </ul>
        """
        impute_html += imputation_recs_to_html(recs_df)

        # If comparison mode, show cleaned dataset recommendations
        if comparison_mode:
            df_cleaned = df.copy()
            for _, row in recs_df.iterrows():
                col = row["Column"]
                if "Mean" in row["Suggestions"]:
                    df_cleaned[col] = df_cleaned[col].fillna(df[col].mean(skipna=True))
                elif "Median" in row["Suggestions"]:
                    df_cleaned[col] = df_cleaned[col].fillna(df[col].median(skipna=True))
                elif "Most Frequent" in row["Suggestions"]:
                    mode_val = df[col].mode(dropna=True)
                    if not mode_val.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val.iloc[0])

            cleaned_recs_df = get_imputation_recommendations(df_cleaned)
            impute_html += "<h3>Cleaned Dataset</h3>"
            impute_html += """
            <ul>
                <li>Displays updated strategies after imputations applied.</li>
                <li>Confirms whether cleaning fully resolved missing values.</li>
            </ul>
            """
            if cleaned_recs_df.empty or cleaned_recs_df["Suggestions"].eq("").all():
                impute_html += "<p><b>All missing values resolved — no further imputations needed.</b></p>"
            else:
                impute_html += imputation_recs_to_html(cleaned_recs_df)

        missing_html += impute_html
    except Exception as e:
        print(f"[IMPUTATION] Error generating recommendations: {e}")
        missing_html += f"<h2>Imputation Recommendations</h2><p>Error: {e}</p>"

    # ----------- Schema Export Section -----------
    try:
        schema_html = "<h2>Schema Export</h2>"
        schema_html += """
        <p><i>Why this matters:</i></p>
        <ul>
            <li>Think of it as the <b>blueprint of the dataset</b> — without it, you'd have to trust memory or rely on fragile assumptions about the CSV.</li>
            <li>It provides an <b>audit trail</b> for comparing runs and spotting schema drift over time.</li>
            <li>Analysts can quickly validate if new data <b>matches expectations</b> before running deeper analysis or training models.</li>
        </ul>
        """

        # Always export original schema (suffix-safe)
        base_html = Path(output_path)
        schema_original = export_schema_yaml(
            df,
            str(base_html.with_suffix("")) + "-original.schema.expected.yaml"
        )
        schema_html += "<h3>Original Dataset Schema</h3>"
        schema_html += f"<p>Schema file exported: {schema_original}</p>"

        # Fix: drop only .html suffix so .csv stays intact
        original_csv_path = str(base_html.with_suffix(""))  
        schema_html += f"<p>Original dataset saved to: {original_csv_path}</p>"

        # If comparison mode, also export cleaned schema (suffix-safe)
        if comparison_mode:
            df_cleaned = df.copy()
            recs_df = get_imputation_recommendations(df)
            for _, row in recs_df.iterrows():
                col = row["Column"]
                if "Mean" in row["Suggestions"]:
                    df_cleaned[col] = df_cleaned[col].fillna(df[col].mean(skipna=True))
                elif "Median" in row["Suggestions"]:
                    df_cleaned[col] = df_cleaned[col].fillna(df[col].median(skipna=True))
                elif "Most Frequent" in row["Suggestions"]:
                    mode_val = df[col].mode(dropna=True)
                    if not mode_val.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val.iloc[0])

            schema_cleaned = export_schema_yaml(
                df_cleaned,
                str(base_html.with_suffix("")) + "-cleaned.schema.expected.yaml"
            )
            schema_html += "<h3>Cleaned Dataset Schema</h3>"
            schema_html += f"<p>Schema file exported: {schema_cleaned}</p>"

            # Fix: clean up duplicate .csv.csv
            cleaned_csv_path = str(
                Path(output_path).with_name(
                    "CLEANED_" + Path(output_path).stem.replace(".csv", "") + ".csv"
                )
            )
            schema_html += f"<p>Cleaned dataset saved to: {cleaned_csv_path}</p>"

        else:
            schema_html += "<h3>Cleaned Dataset Schema</h3>"
            schema_html += (
                "<p><em>Cleaned dataset schema not generated. "
                "Run with comparison mode enabled to export both original and cleaned schemas.</em></p>"
            )
            print("[SCHEMA] Cleaned dataset schema not generated. "
                "Run with comparison mode enabled to export both original and cleaned schemas.")

        missing_html += schema_html
    except Exception as e:
        missing_html += f"<h2>Schema Export</h2><p>Error exporting schema: {e}</p>"

    # ----------- Inline Schema Comparison (HTML diff) -----------
    try:
        if comparison_mode:
            schema_diff_html = "<h3>Schema Differences (Columns & Data Types)</h3>"

            # Collect column/dtype info
            original_schema = pd.DataFrame({
                "Column": df.columns,
                "Dtype": [str(df[col].dtype) for col in df.columns]
            })
            cleaned_schema = pd.DataFrame({
                "Column": df_cleaned.columns,
                "Dtype": [str(df_cleaned[col].dtype) for col in df_cleaned.columns]
            })

            # Merge on column names
            merged = pd.merge(
                original_schema,
                cleaned_schema,
                on="Column",
                how="outer",
                suffixes=("_Original", "_Cleaned")
            ).fillna("MISSING")

            # Bold changes for clarity
            for idx, row in merged.iterrows():
                if row["Dtype_Original"] != row["Dtype_Cleaned"]:
                    merged.at[idx, "Dtype_Cleaned"] = f"<b>{row['Dtype_Cleaned']}</b>"
                if row["Dtype_Original"] == "MISSING":
                    merged.at[idx, "Dtype_Original"] = "<b>MISSING</b>"
                if row["Dtype_Cleaned"] == "MISSING":
                    merged.at[idx, "Dtype_Cleaned"] = "<b>MISSING</b>"

            schema_diff_html += merged.to_html(classes="table table-striped", index=False, escape=False)
            missing_html += schema_diff_html
    except Exception as e:
        missing_html += f"<h3>Schema Differences</h3><p>Error generating schema diff: {e}</p>"

    # ----------- Duplicate Detection -----------
    try:
        print("[DUPLICATES] Starting duplicate detection...")

        dup_count = df.duplicated().sum()
        total_rows = len(df)
        pct_dup = (dup_count / total_rows) * 100 if total_rows > 0 else 0

        def assign_dup_flag(pct: float) -> str:
            g, y = get_cutoffs("Duplicate Rows", 1, 5)  # Green <1%, Yellow 1–5%, Red >5%
            if pct < g:
                return "Green"
            elif pct <= y:
                return "Yellow"
            else:
                return "Red"

        severity = assign_dup_flag(pct_dup)
        severity_colored = colorize_severity(severity)

        dup_html = f"""
        <h2>Duplicate Rows</h2>
        <p>Total Duplicates: {dup_count} ({pct_dup:.2f}%)</p>
        <p>Severity: {severity_colored}</p>
        <ul>
            <li><b>Green:</b> Duplicates &lt;1% → minimal concern.</li>
            <li><b>Yellow:</b> Duplicates 1–5% → review advised.</li>
            <li><b>Red:</b> Duplicates &gt;5% → strong concern, remove or investigate source data.</li>
        </ul>
        """
        print(f"[DUPLICATES] Completed successfully: {dup_count} duplicates found ({pct_dup:.2f}%).")
    except Exception as e:
        print(f"[DUPLICATES] Error during duplicate detection: {e}")
        dup_html = f"<h2>Duplicate Rows</h2><p>Error: {e}</p>"

    # ----------- Outlier Detection (Z-Score + IQR) -----------
    try:
        print("[OUTLIERS] Starting outlier detection...")

        outlier_z = detect_outliers_zscore(df)
        outlier_iqr = detect_outliers_iqr(df)

        # Apply severity logic
        def assign_outlier_flag(pct: float) -> str:
            g, y = get_cutoffs("Outliers", 5, 15)  # Green <5%, Yellow 5–15%, Red >15%
            if pct < g:
                return "Green"
            elif pct <= y:
                return "Yellow"
            else:
                return "Red"

        for df_out in [outlier_z, outlier_iqr]:
            if not df_out.empty:
                df_out["Severity"] = df_out["Pct_Outliers"].apply(assign_outlier_flag)
                df_out["Severity"] = df_out["Severity"].apply(colorize_severity)

        outlier_html = """
        <h2>Outlier Detection</h2>
        <p><i>Outliers are unusual values that may distort analysis. Severity coloring helps prioritize attention:</i></p>
        <ul>
            <li><b>Green:</b> Distribution is stable, few anomalies detected.</li>
            <li><b>Yellow:</b> Moderate anomalies; investigate possible data issues.</li>
            <li><b>Red:</b> High anomaly rate; strong indication of outliers or quality problems.</li>
        </ul>
        <p><i>Best Practice: % Outliers &lt;5% = Green, 5–15% = Yellow, &gt;15% = Red</i></p>
        <ul>
            <li>Outliers below 5% are considered minimal and generally safe.</li>
            <li>Outliers between 5–15% warrant review; may indicate data quality issues.</li>
            <li>Outliers above 15% are high risk; strong chance of data entry errors or unusual distributions.</li>
        </ul>
        """

        if not outlier_z.empty:
            outlier_html += "<h3>Z-Score Method</h3>"
            outlier_html += outlier_z.to_html(classes="table table-striped", index=False, escape=False)
            outlier_html += """
            <ul>
                <li>Z-Score formula: Z = (x − μ) / σ</li>
                <li>Flags points far from the mean; best for symmetric distributions.</li>
            </ul>
            """
        else:
            outlier_html += "<h3>Z-Score Method</h3><p>No numeric columns or no outliers detected.</p>"

        if not outlier_iqr.empty:
            outlier_html += "<h3>IQR Method</h3>"
            outlier_html += outlier_iqr.to_html(classes="table table-striped", index=False, escape=False)
            outlier_html += """
            <ul>
                <li>IQR formula: [Q1 − 1.5×IQR, Q3 + 1.5×IQR]</li>
                <li>Robust for skewed distributions or datasets with extreme values.</li>
            </ul>
            """
        else:
            outlier_html += "<h3>IQR Method</h3><p>No numeric columns or no outliers detected.</p>"

        print("[OUTLIERS] Outlier detection completed successfully.")
    except Exception as e:
        print(f"[OUTLIERS] Error during outlier detection: {e}")
        outlier_html = f"<h2>Outlier Detection</h2><p>Error: {e}</p>"

    # ----------- Class Imbalance Detection -----------
    try:
        print("[IMBALANCE] Starting class imbalance analysis...")

        imbalance_html = """
        <h2>Class Imbalance</h2>
        <p><i>Imbalanced classes can bias models toward the majority class, reducing accuracy on minority classes.</i></p>
        <p><i>Best Practice: Balance or mitigate imbalance before training predictive models.</i></p>
        <ul>
            <li><b>Green:</b> No severe imbalance, safe for most models.</li>
            <li><b>Yellow:</b> Moderate imbalance, monitor during modeling.</li>
            <li><b>Red:</b> Severe imbalance, apply resampling or advanced techniques.</li>
        </ul>
        <p><i>Note: Columns with >50 unique values or >20% unique ratio are automatically skipped as they likely represent IDs, timestamps, or high-cardinality features not suitable for imbalance analysis.</i></p>
        """

        cat_cols = df.select_dtypes(include=["object", "category", "datetime64[ns]"]).columns
        total_rows = len(df)
        skipped_count = 0
        analyzed_count = 0
        
        for col in cat_cols:
            unique_count = df[col].nunique(dropna=True)
            unique_ratio = unique_count / total_rows if total_rows > 0 else 0
            
            # Enhanced skip logic: datetime types, high cardinality (>50), or high unique ratio (>20%)
            # Also check column name patterns that suggest datetime/ID columns
            col_lower = str(col).lower()
            is_likely_datetime_or_id = any(pattern in col_lower for pattern in 
                ['date', 'time', 'timestamp', 'datetime', 'id', 'key', 'uuid', 'guid'])
            
            should_skip = (
                pd.api.types.is_datetime64_any_dtype(df[col]) or 
                unique_count > 50 or 
                unique_ratio > 0.2 or
                (is_likely_datetime_or_id and unique_count > 20)
            )
            
            if should_skip:
                skipped_count += 1
                imbalance_html += f"<h3>{col}</h3>"
                skip_reason = []
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    skip_reason.append("datetime type")
                if unique_count > 50:
                    skip_reason.append(f"{unique_count:,} unique values")
                if unique_ratio > 0.2:
                    skip_reason.append(f"{unique_ratio:.1%} unique ratio")
                if is_likely_datetime_or_id and unique_count > 20:
                    skip_reason.append("likely datetime/ID column")
                
                imbalance_html += f"<p><em>Skipped: {', '.join(skip_reason)}</em></p>"
                continue

            # Analyze columns with reasonable cardinality
            value_counts = df[col].value_counts(normalize=True) * 100
            if len(value_counts) > 1:
                analyzed_count += 1
                max_class_pct = value_counts.iloc[0]

                if max_class_pct < 70:
                    severity = "Green"
                elif max_class_pct <= 90:
                    severity = "Yellow"
                else:
                    severity = "Red"

                severity_colored = colorize_severity(severity)

                imbalance_html += f"<h3>{col}</h3>"
                imbalance_html += f"<p>Largest class: {max_class_pct:.2f}% of records → Severity: {severity_colored}</p>"
                imbalance_html += f"<p><em>{unique_count} unique categories</em></p>"
                
                # Show only top 10 categories
                top_n = 10
                imbalance_html += value_counts.head(top_n).to_frame("Percentage").to_html(
                    classes="table table-striped", border=0
                )
                if unique_count > top_n:
                    imbalance_html += f"<p><em>Showing top {top_n} of {unique_count} categories.</em></p>"

        # Summary at the end
        if analyzed_count == 0 and skipped_count > 0:
            imbalance_html += "<p><strong>No suitable categorical columns found for imbalance analysis. All columns were skipped due to high cardinality or datetime/ID patterns.</strong></p>"
        
        print(f"[IMBALANCE] Class imbalance analysis completed: {analyzed_count} analyzed, {skipped_count} skipped.")
    except Exception as e:
        print(f"[IMBALANCE] Error during class imbalance analysis: {e}")
        imbalance_html = f"<h2>Class Imbalance</h2><p>Error: {e}</p>"

    # ----------- Correlation / Multicollinearity -----------
    try:
        print("[CORRELATION] Starting correlation analysis...")

        corr_html = "<h2>Correlation Heatmap</h2><p>Not enough numeric columns</p>"
        numeric_cols = plot_df.select_dtypes(include="number").columns

        if len(numeric_cols) > 1:
            corr_matrix = plot_df[numeric_cols].corr()
            flagged = []
            g, y = get_cutoffs("Correlation", 85, 90)  # Green <85%, Yellow 85–90%, Red >90%

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    r = abs(corr_matrix.iloc[i, j]) * 100
                    if r < g:
                        severity = "Green"
                    elif r <= y:
                        severity = "Yellow"
                    else:
                        severity = "Red"
                    flagged.append({
                        "Var1": numeric_cols[i],
                        "Var2": numeric_cols[j],
                        "Correlation %": round(r, 2),
                        "Severity": severity
                    })

            flagged_df = pd.DataFrame(flagged).sort_values(by="Correlation %", ascending=False)

            if not flagged_df.empty:
                flagged_df["Severity"] = flagged_df["Severity"].apply(colorize_severity)
                corr_html = """
                <h2>Correlation Analysis</h2>
                <p><i>Best Practice: |r| &lt;85% = Green, 85–90% = Yellow, &gt;90% = Red</i></p>
                <ul>
                    <li>Highlights strength of linear relationships between variables.</li>
                    <li>Helps detect redundancy and multicollinearity issues.</li>
                    <li>Guides feature selection before modeling.</li>
                </ul>
                <ul>
                    <li><b>Green:</b> Acceptable correlations, minimal redundancy.</li>
                    <li><b>Yellow:</b> Moderate correlations, monitor for multicollinearity issues.</li>
                    <li><b>Red:</b> High correlation; likely multicollinearity, remove or combine features.</li>
                </ul>
                """ + flagged_df.to_html(classes="table table-striped", index=False, escape=False)

            # Generate and embed heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plots_dir = Path(output_path).parent / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            corr_path = plots_dir / "corr_heatmap.png"
            plt.savefig(corr_path, bbox_inches="tight")
            plt.close()
            chart_b64 = encode_image_base64(corr_path)
            corr_html += f'<div><img src="{chart_b64}" alt="Correlation Heatmap"></div>'

            print("[CORRELATION] Correlation analysis completed successfully.")
        else:
            print("[CORRELATION] Skipped — not enough numeric columns for correlation analysis.")

    except Exception as e:
        print(f"[CORRELATION] Error during correlation analysis: {e}")
        corr_html = f"<h2>Correlation Analysis</h2><p>Error: {e}</p>"

    # ----------- Univariate Analysis -----------
    try:
        print("[UNIVARIATE] Starting univariate analysis...")

        univariate_html = """
        <h2>Univariate Analysis</h2>
        <p>Univariate analysis examines one variable at a time to understand its distribution:</p>
        <ul>
            <li>Numeric features: histograms with KDE overlay to show spread and skewness.</li>
            <li>Categorical features: bar plots of top categories.</li>
            <li>Helps detect imbalance, skew, and unusual distributions.</li>
        </ul>
        <table><tr>
        """

        col_count = 0
        plots_dir = Path(output_path).parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for col in df.columns:
            try:
                plt.figure(figsize=(6, 4))
                if pd.api.types.is_numeric_dtype(df[col]):
                    sns.histplot(df[col].dropna(), kde=True)
                    plt.title(f"Distribution of {col}")
                    plt.xticks(rotation=45, ha="right")
                else:
                    df[col].value_counts().head(20).plot(kind="bar")
                    plt.title(f"Top Categories in {col}")
                    plt.xticks(rotation=45, ha="right")

                uni_path = plots_dir / f"uni_{sanitize_filename(col)}.png"
                plt.tight_layout()
                plt.savefig(uni_path, bbox_inches="tight")
                plt.close()

                # Embed as a table cell
                univariate_html += (
                    f'<td style="padding:10px; text-align:center;">'
                    f"<h3>{col}</h3>"
                    f'<img src="{encode_image_base64(uni_path)}" alt="Univariate {col}" style="max-width:500px;">'
                    "</td>"
                )

            except Exception as inner_e:
                print(f"[UNIVARIATE] Skipped column {col} due to error: {inner_e}")
                univariate_html += (
                    f'<td style="padding:10px; text-align:center;">'
                    f"<h3>{col}</h3><p>[WARNING] Could not generate plot: {inner_e}</p>"
                    "</td>"
                )

            col_count += 1
            if col_count % 3 == 0 and col_count != len(df.columns):
                univariate_html += "</tr><tr>"

        univariate_html += "</tr></table>"
        print("[UNIVARIATE] Univariate analysis completed successfully.")

    except Exception as e:
        print(f"[UNIVARIATE] Error during univariate analysis: {e}")
        univariate_html = f"<h2>Univariate Analysis</h2><p>Error: {e}</p>"

    # ----------- Multivariate Analysis -----------
    multivariate_html = "<h2>Multivariate Analysis</h2>"
    multivariate_html += """
    <ul>
        <li>Examines interactions between two or more variables.</li>
        <li>Helps analysts detect patterns not visible in univariate views.</li>
        <li>Supports feature engineering and hypothesis testing.</li>
    </ul>
    """
    numeric_cols = plot_df.select_dtypes(include="number").columns
    if len(numeric_cols) > 1:
        # Cap pairplot at 500 rows to avoid performance issues
        capped_df = plot_df[numeric_cols].sample(n=min(len(plot_df), 500), random_state=42)
        sns.pairplot(capped_df)
        pairplot_path = plots_dir / "pairplot.png"; plt.savefig(pairplot_path); plt.close()
        multivariate_html += f'<div><img src="{encode_image_base64(pairplot_path)}" alt="Pairplot"></div>'

    try:
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        nlp_html = """
        <h2>Natural Language Processing</h2>
        <p>This section provides insights into text data quality and content.</p>
        <ul>
            <li>Tokenization of text fields (splitting into words)</li>
            <li>Stopword removal and stemming</li>
            <li>Word frequency distributions</li>
        </ul>
        """
        nlp_html += nlp_word_frequencies(df, text_columns=text_cols, top_n=20)
    except Exception as e:
        print(f"[ERROR] Multivariate or NLP analysis failed: {e}")  # console log
        multivariate_html = f"<p>[ERROR] Multivariate analysis failed: {e}</p>"
        nlp_html = f"<p>[ERROR] NLP analysis failed: {e}</p>"

    # ----------- HTML Template -----------
    template_str = """
    <html>
    <body>
        <h1>Data Quality Analyzer Report</h1>
        {{ summary | safe }}
        {{ missing | safe }}
        {{ dup | safe }}
        {{ outliers | safe }}
        {{ imbalance | safe }}
        {{ corr | safe }}
        {{ univariate | safe }}
        {{ multivariate | safe }}
        {{ nlp | safe }}
        <br><br>
        <footer style="position: fixed; bottom: 10px; left: 10px; font-size: 12px; color: gray;">
            DQA Report - Brock Frary - v2.7.0
        </footer>
    </body>
    </html>
    """
    template = Template(template_str)

    # ----------- Render HTML Content -----------
    try:
        print("[REPORT] Assembling final HTML report...")
        html_content = template.render(
            summary=summary_html,
            missing=missing_html,
            dup=dup_html,
            outliers=outlier_html,
            imbalance=imbalance_html,
            corr=corr_html,
            univariate=univariate_html,
            multivariate=multivariate_html,
            nlp=nlp_html,
        )
        print("[REPORT] HTML report assembled successfully.")
    except Exception as e:
        print(f"[REPORT] Error assembling HTML report: {e}")
        html_content = f"<h1>Data Quality Analyzer Report</h1><p>Error: {e}</p>"

    # Export schema YAML (original schema only, suffix-safe)
    base_html = Path(output_path)
    schema_path = export_schema_yaml(
        df,
        str(base_html.with_suffix("")) + "-original.schema.expected.yaml"
    )

    # ----------- Write HTML Report -----------
    try:
        # print("### DEBUG HTML WRITE ### Attempting to write:", output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        # print(f"[DEBUG] HTML report write complete.")
        print(f"[REPORT] HTML successfully written to: {output_path}")
        if not Path(output_path).exists():
            print(f"[ERROR] Expected HTML file not found after write: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write HTML file: {e}")

    # ----------- PDF Export (Optional via Console Toggle) -----------
    try:
        choice = input("Generate PDF version of the report? (Y/N) [default: N]: ").strip().lower()
        if choice == "y":
            pdf_path = str(Path(output_path).with_suffix(".pdf"))
            from dqa_utils import export_pdf_from_html
            export_pdf_from_html(output_path, pdf_path)
        else:
            print("[EXPORT] PDF export skipped.")
    except Exception as e:
        print(f"[EXPORT] Error creating PDF: {e}")

    # Ensure df_cleaned is always defined for return
    if comparison_mode:
        return str(output_path), df_cleaned
    else:
        return str(output_path), df


# ----------- Get Imputation Recommendations -----------
def get_imputation_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate imputation recommendations for columns with missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze for missing values.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing imputation recommendations with columns:
        - Column: column name
        - Type: data type
        - Missing %: percentage of missing values
        - Suggestions: recommended imputation strategy
    """
    recs = []
    
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct == 0:
            continue

        col_type = df[col].dtype
        strategy = "N/A"

        # Numeric strategy
        if pd.api.types.is_numeric_dtype(col_type):
            if missing_pct < 5:
                strategy = "Mean or Median Imputation"
            elif missing_pct <= 30:
                strategy = "Predictive/Model-Based Imputation"
            else:
                strategy = "Drop Column (too many missing)"

        # Categorical strategy
        elif pd.api.types.is_categorical_dtype(col_type) or df[col].dtype == object:
            if missing_pct < 5:
                strategy = "Mode Imputation"
            elif missing_pct <= 30:
                strategy = "Group-Based Imputation"
            else:
                strategy = "Drop Column (too many missing)"

        # Datetime strategy
        elif np.issubdtype(col_type, np.datetime64):
            if missing_pct < 5:
                strategy = "Forward/Backward Fill"
            elif missing_pct <= 30:
                strategy = "Interpolation"
            else:
                strategy = "External Data / Drop Column"

        else:
            strategy = "Manual Review Required"

        recs.append({
            "Column": col,
            "Type": str(col_type),
            "Missing %": round(missing_pct, 2),
            "Suggestions": strategy
        })

    return pd.DataFrame(recs)


# ----------- Imputation Recommendations to HTML ----------- 
def imputation_recs_to_html(recs_df: pd.DataFrame) -> str:
    """
    Convert imputation recommendations DataFrame to HTML.
    Includes explanatory notes if table is empty.
    """
    if recs_df.empty:
        return "<p>No missing values detected. No imputations required.</p>"

    return recs_df.to_html(index=False, classes="dataframe table table-striped", border=1, escape=False)


# ----------- PDF Export (HTML → PDF via pdfkit) -----------
def export_pdf_from_html(html_path: str, pdf_path: str):
    """
    Convert an existing HTML report into a PDF using pdfkit + wkhtmltopdf.
    Ensures charts, tables, and CSS styling are preserved exactly
    as seen in the HTML export.
    """
    import pdfkit
    try:
        print("[EXPORT] Converting HTML to PDF (via pdfkit)...")
        config = pdfkit.configuration(wkhtmltopdf=r"D:\wkhtmltopdf\bin\wkhtmltopdf.exe")

        # Options for better fidelity with HTML
        options = {
            "enable-local-file-access": "",   # Allow local CSS/images
            "quiet": "",                      # Suppress wkhtmltopdf console spam
            "encoding": "UTF-8",
            "page-size": "Letter",
            "margin-top": "0.5in",
            "margin-bottom": "0.5in",
            "margin-left": "0.5in",
            "margin-right": "0.5in",
        }

        pdfkit.from_file(html_path, pdf_path, options=options, configuration=config)
        print(f"[EXPORT] PDF created: {pdf_path}")
    except Exception as e:
        print(f"[EXPORT] Error during HTML-to-PDF conversion: {e}")

def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers using Z-Score method.
    Outliers are values more than `threshold` standard deviations from the mean.
    """
    results = []
    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            # No variation → no outliers
            results.append({"Column": col, "Method": "Z-Score", "Num_Outliers": 0, "Pct_Outliers": 0.0})
            continue

        z_scores = (series - mean) / std
        outliers = (np.abs(z_scores) > threshold).sum()
        pct_outliers = (outliers / len(series)) * 100
        results.append({
            "Column": col,
            "Method": "Z-Score",
            "Num_Outliers": int(outliers),
            "Pct_Outliers": round(pct_outliers, 2)
        })

    return pd.DataFrame(results)

# ----------- Helper: Outlier Detection (IQR) -----------
def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    Outliers are values outside [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    results = []
    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0 or np.isnan(IQR):
            # No spread → no outliers
            results.append({"Column": col, "Method": "IQR", "Num_Outliers": 0, "Pct_Outliers": 0.0})
            continue

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outliers = ((series < lower) | (series > upper)).sum()
        pct_outliers = (outliers / len(series)) * 100
        results.append({
            "Column": col,
            "Method": "IQR",
            "Num_Outliers": int(outliers),
            "Pct_Outliers": round(pct_outliers, 2)
        })

    return pd.DataFrame(results)

# ----------- Export Schema as YAML -----------
def export_schema_yaml(df: pd.DataFrame, output_path: str):
    """
    Export schema information (columns, dtypes, nullability, ranges, categories) to a YAML file.
    """
    schema = {"columns": []}
    for col in df.columns:
        col_info = {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isnull().any())
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min(skipna=True)) if not df[col].dropna().empty else None
            col_info["max"] = float(df[col].max(skipna=True)) if not df[col].dropna().empty else None
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 50:  # limit to avoid giant lists
                col_info["categories"] = [str(v) for v in unique_vals]
        schema["columns"].append(col_info)

    # print(f"[DEBUG] Schema being exported to: {output_path}") 
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    # print(f"[DEBUG] Schema export complete.") 
    return str(output_path)

# ----------- NLP Utilities -----------
def preprocess_text(text: str, perform_nlp: bool = True) -> list:
    """
    Preprocess a text string into tokens.

    Parameters
    ----------
    text : str
        The input text string.
    perform_nlp : bool
        If True, apply stopword removal + stemming.
        If False, only lowercase and strip punctuation.

    Returns
    -------
    list
        List of processed tokens.
    """
    # Always lowercase + remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()

    if not perform_nlp:
        # Only basic cleaning
        return tokens

    # Full NLP: stopword removal + stemming
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words]

    return tokens


def tokenize_and_filter(texts: list) -> list:
    """
    Apply text preprocessing to a list of documents.

    Iterates over a list of text entries (e.g., dataset column values) 
    and applies preprocess_text() to each valid string. Non-string 
    values are ignored.

    Parameters
    ----------
    texts : list
        A list of text entries such as reviews, comments, or 
        free-text fields.

    Returns
    -------
    list
        A list of lists, where each inner list contains tokens 
        (normalized and stemmed words) for a single document.

    Notes
    -----
    - Designed for batch preprocessing of textual columns in datasets.
    - Typically used before word frequency analysis or NLP visualization.
    """
    return [preprocess_text(t) for t in texts if isinstance(t, str)]


# ----------- NLP Word Frequency Analysis -----------
from collections import Counter

def nlp_word_frequencies(
    df: pd.DataFrame, text_columns: list, top_n: int = 20, perform_nlp: bool = True
) -> str:
    """
    Generate word frequency tables for specified text columns.

    For each given column containing free-text data:
    - Builds two frequency distributions:
        * Original tokens (lowercased, whitespace split).
        * Cleaned tokens (basic cleaning only, or full NLP if perform_nlp=True).
    - Returns as side-by-side HTML tables.

    Layout:
    - 2 features per row (6 columns total: 3 per feature).
    - Each feature displays: Feature Name | Original Tokens | Cleaned Tokens.
    - Main table has no borders; inner token tables keep borders for readability.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing text columns.
    text_columns : list
        List of column names in df to analyze.
    top_n : int
        Number of most frequent tokens to return per column.
    perform_nlp : bool
        If True, apply stopword removal + stemming.
        If False, only lowercase and strip punctuation.

    Returns
    -------
    str
        HTML string with formatted NLP frequency tables.
    """
    try:
        print("[NLP] Starting NLP word frequency analysis...")

        # Explanatory bullets only (header is handled in multivariate_html)
        mode_note = (
            "<b>NLP performed:</b> Full (stopwords + stemming)"
            if perform_nlp else
            "<b>NLP performed:</b> Basic Cleaning Only (lowercasing, punctuation removal)"
        )

        html_output = f"""
        <ul>
            <li><b>Original text:</b> keeps analyst context (stopwords, punctuation, surface forms).</li>
            <li><b>Cleaned text:</b> normalized tokens prepared for modeling.</li>
            <li>Side-by-side view helps analysts validate preprocessing.</li>
            <li>{mode_note}</li>
        </ul>
        <table style="border:none; border-collapse:collapse;"><tr>
        """
        
        feature_count = 0
        for col in text_columns:
            if col not in df.columns:
                continue

            # Start new row every 2 features
            if feature_count > 0 and feature_count % 2 == 0:
                html_output += "</tr><tr>"

            # Collect raw/original tokens (always lowercase + strip punctuation only)
            raw_tokens = []
            for val in df[col].dropna():
                text = str(val).lower()
                text = re.sub(r"[^a-z0-9\s]", " ", text)
                raw_tokens.extend(text.split())

            # Collect cleaned tokens (basic or full NLP depending on perform_nlp flag)
            cleaned_tokens = []
            for val in df[col].dropna():
                cleaned_tokens.extend(preprocess_text(str(val), perform_nlp=perform_nlp))

            # Frequency DataFrames
            raw_freq_df = (
                pd.DataFrame(Counter(raw_tokens).most_common(top_n), columns=["Token", "Count"])
                if raw_tokens else pd.DataFrame(columns=["Token", "Count"])
            )
            clean_freq_df = (
                pd.DataFrame(Counter(cleaned_tokens).most_common(top_n), columns=["Token", "Count"])
                if cleaned_tokens else pd.DataFrame(columns=["Token", "Count"])
            )

            # Feature name cell
            html_output += (
                f"<td style='padding:10px; vertical-align:top; text-align:center; border:none;'>"
                f"<h3>{col}</h3></td>"
            )
            
            # Original Tokens cell
            html_output += (
                f"<td style='padding:10px; vertical-align:top; border:none;'>"
                f"<b>Original Tokens</b>{raw_freq_df.to_html(index=False, classes='table table-striped', border=1)}"
                f"</td>"
            )
            
            # Cleaned Tokens cell
            html_output += (
                f"<td style='padding:10px; vertical-align:top; border:none;'>"
                f"<b>Cleaned Tokens</b>{clean_freq_df.to_html(index=False, classes='table table-striped', border=1)}"
                f"</td>"
            )

            feature_count += 1

        html_output += "</tr></table>"

        print("[NLP] NLP word frequency analysis completed successfully.")
        return html_output

    except Exception as e:
        print(f"[NLP] Error during NLP analysis: {e}")
        return f"<h2>Natural Language Processing</h2><p>Error: {e}</p>"


# © 2025 Brock Frary. All rights reserved.