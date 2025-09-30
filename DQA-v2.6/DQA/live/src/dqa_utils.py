
# Developer: Brock Frary
# Title: dqa_utils.py
# Date: 09/27/2025
# Version: 1.0.2 (restored full functionality with best practices, color coding, and charts)

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
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# ----------- Helper: Sanitize Filenames -----------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', "_", str(name))

# ----------- Helper: Threshold Assignment -----------
def get_cutoffs(check_name: str, default_green: int, default_yellow: int):
    green_cutoff, yellow_cutoff = default_green, default_yellow
    try:
        row = thresholds_df[thresholds_df["Check"].str.contains(check_name, case=False)].iloc[0]
        rule_text = str(row["Threshold / Rule of Thumb"])
        nums = re.findall(r"\d+", rule_text)
        if len(nums) >= 2:
            green_cutoff = int(nums[0])
            yellow_cutoff = max(int(n) for n in nums[1:])
    except Exception:
        pass
    return green_cutoff, yellow_cutoff

# ----------- Helper: Colorize Severity -----------
def colorize_severity(label: str) -> str:
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
    html_parts = []

    html_parts.append("<h2>DataFrame Shape</h2>")
    html_parts.append(
        "<ul>"
        "<li>Indicates dataset size. Ensure enough rows for robust analysis, but not so many that compute becomes inefficient.</li>"
        "<li>Columns should balance completeness and usability—too few may omit important context, too many may introduce noise.</li>"
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
    html_parts.append(
        "<ul>"
        "<li>Displays the first few records to verify headers and values.</li>"
        "<li>Helps detect misaligned columns or unexpected formats early.</li>"
        "</ul>"
    )
    html_parts.append(df.head(n_rows).to_html(classes="table table-striped", border=0))

    html_parts.append("<h2>Last Rows (tail)</h2>")
    html_parts.append(
        "<ul>"
        "<li>Displays the dataset’s final rows to check for trailing issues.</li>"
        "<li>Useful for spotting incomplete records or file truncation.</li>"
        "</ul>"
    )
    html_parts.append(df.tail(n_rows).to_html(classes="table table-striped", border=0))

    try:
        html_parts.append("<h2>Descriptive Statistics (Numeric Columns)</h2>")
        html_parts.append(
            "<ul>"
            "<li>Summarizes mean, median, spread, and extreme values.</li>"
            "<li>Check for unrealistic ranges or skewness that may indicate errors.</li>"
            "</ul>"
        )
        html_parts.append(df.describe(include=[float, int]).to_html(classes="table table-striped", border=0))
    except Exception as e:
        html_parts.append(f"<p><i>No numeric columns found. ({e})</i></p>")

    try:
        html_parts.append("<h2>Descriptive Statistics (Categorical Columns)</h2>")
        html_parts.append(
            "<ul>"
            "<li>Summarizes category frequencies.</li>"
            "<li>Look for dominant values, unexpected rare labels, or excessive unique categories (possible IDs).</li>"
            "</ul>"
        )
        html_parts.append(df.describe(include=[object]).to_html(classes="table table-striped", border=0))
    except Exception as e:
        html_parts.append(f"<p><i>No categorical columns found. ({e})</i></p>")

    return "\n".join(html_parts)

# ----------- Export HTML Report -----------
def export_html_report(df: pd.DataFrame, output_path="dst/dqa_report.html", sample_size: int = 1000):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary_html = summarize_dataframe_html(df)
    plot_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    plots_dir = Path(output_path).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ----------- Missing Values -----------
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing %": missing_pct.round(2)
    }).sort_values(by="Missing Count", ascending=False)

    g, y = get_cutoffs("Missing Values", 5, 30)
    def assign_missing_flag(pct: float) -> str:
        if pct < g: return "Green"
        elif pct <= y: return "Yellow"
        else: return "Red"
    missing_df["Severity"] = missing_df["Missing %"].apply(assign_missing_flag)
    missing_df["Severity"] = missing_df["Severity"].apply(colorize_severity)

    missing_html = """
    <h2>Missing Values per Column</h2>
    <p><i>Best Practice: &lt;5% = Green, 5–30% = Yellow, &gt;30% = Red</i></p>
    <ul>
        <li><b>Green:</b> Data completeness is good, minimal cleaning needed.</li>
        <li><b>Yellow:</b> Moderate missingness; imputation or domain-specific handling required.</li>
        <li><b>Red:</b> High missingness, column may be unreliable or require removal.</li>
    </ul>
    """ + missing_df.to_html(classes="table table-striped", index=False, escape=False)

    # ----------- Duplicate Rows -----------
    dup_count = df.duplicated().sum()
    dup_pct = round((dup_count / len(df)) * 100, 2)
    dup_df = pd.DataFrame({"Duplicate Count": [dup_count], "Duplicate %": [dup_pct]})
    g, y = get_cutoffs("Duplicate Rows", 1, 5)
    def assign_dup_flag(pct: float) -> str:
        if pct < g: return "Green"
        elif pct <= y: return "Yellow"
        else: return "Red"
    dup_df["Severity"] = dup_df["Duplicate %"].apply(assign_dup_flag)
    dup_df["Severity"] = dup_df["Severity"].apply(colorize_severity)
    dup_html = """
    <h2>Duplicate Rows</h2>
    <p><i>Best Practice: &lt;1% = Green, 1–5% = Yellow, &gt;5% = Red</i></p>
    <ul>
        <li><b>Green:</b> Low duplication; data quality is strong.</li>
        <li><b>Yellow:</b> Noticeable duplication; may inflate counts or bias analysis.</li>
        <li><b>Red:</b> Excessive duplication; must be resolved before reliable use.</li>
    </ul>
    """ + dup_df.to_html(classes="table table-striped", index=False, escape=False)

    # ----------- Outlier Detection (Z-score method) -----------
    outlier_results = []
    num_cols = df.select_dtypes(include=np.number).columns
    g, y = get_cutoffs("Outliers", 5, 10)
    for col in num_cols:
        series = df[col].dropna()
        if series.empty: continue
        z_scores = (series - series.mean()) / series.std(ddof=0)
        outliers = (abs(z_scores) > 3).sum()
        pct_outliers = round((outliers / len(series)) * 100, 2)
        if pct_outliers < g: severity = "Green"
        elif pct_outliers <= y: severity = "Yellow"
        else: severity = "Red"
        outlier_results.append({"Column": col, "Outlier %": pct_outliers, "Severity": severity})
    outlier_df = pd.DataFrame(outlier_results)
    if not outlier_df.empty:
        outlier_df["Severity"] = outlier_df["Severity"].apply(colorize_severity)
        outlier_html = """
        <h2>Outlier Detection</h2>
        <p><i>Best Practice: &lt;5% outliers = Green, 5–10% = Yellow, &gt;10% = Red</i></p>
        <ul>
            <li><b>Green:</b> Distribution is stable, few outliers detected.</li>
            <li><b>Yellow:</b> Moderate outliers; investigate potential data entry errors or natural variance.</li>
            <li><b>Red:</b> High outlier rate; indicates potential anomalies or quality issues.</li>
        </ul>
        """ + outlier_df.to_html(classes="table table-striped", index=False, escape=False)
    else:
        outlier_html = "<h2>Outlier Detection</h2><p>No numeric columns</p>"

    # ----------- Class Imbalance (for classification-like data) -----------
    imbalance_html = "<h2>Class Imbalance</h2><p>No categorical target detected</p>"
    if "target" in df.columns:
        counts = df["target"].value_counts(normalize=True) * 100
        imbalance_df = counts.reset_index(); imbalance_df.columns = ["Class", "Percent"]
        g, y = get_cutoffs("Class Imbalance", 80, 90)
        imbalance_df["Severity"] = imbalance_df["Percent"].apply(lambda p: "Green" if p < g else ("Yellow" if p <= y else "Red"))
        imbalance_df["Severity"] = imbalance_df["Severity"].apply(colorize_severity)
        imbalance_html = """
        <h2>Class Imbalance</h2>
        <p><i>Best Practice: Majority &lt;80% = Green, 80–90% = Yellow, &gt;90% = Red</i></p>
        <ul>
            <li><b>Green:</b> Classes are well balanced, modeling is reliable.</li>
            <li><b>Yellow:</b> Some imbalance; may require resampling or weighted methods.</li>
            <li><b>Red:</b> Severe imbalance; major corrective action needed before modeling.</li>
        </ul>
        """ + imbalance_df.to_html(classes="table table-striped", index=False, escape=False)

    # ----------- Correlation / Multicollinearity -----------
    corr_html = "<h2>Correlation Heatmap</h2><p>Not enough numeric columns</p>"
    numeric_cols = plot_df.select_dtypes(include="number").columns
    if len(numeric_cols) > 1:
        corr_matrix = plot_df[numeric_cols].corr()
        flagged = []
        g, y = 85, 90
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                r = abs(corr_matrix.iloc[i,j]) * 100
                if r < g: severity = "Green"
                elif r <= y: severity = "Yellow"
                else: severity = "Red"
                flagged.append({"Var1": numeric_cols[i], "Var2": numeric_cols[j], "Correlation %": round(r,2), "Severity": severity})
        flagged_df = pd.DataFrame(flagged)
        if not flagged_df.empty:
            flagged_df["Severity"] = flagged_df["Severity"].apply(colorize_severity)
            corr_html = """
            <h2>Correlation Analysis</h2>
            <p><i>Best Practice: |r| &lt;85% = Green, 85–90% = Yellow, &gt;90% = Red</i></p>
            <ul>
                <li><b>Green:</b> Acceptable correlations, minimal redundancy.</li>
                <li><b>Yellow:</b> Moderate correlations, monitor for multicollinearity issues.</li>
                <li><b>Red:</b> High correlation; likely multicollinearity, remove or combine features.</li>
            </ul>
            """ + flagged_df.to_html(classes="table table-striped", index=False, escape=False)

        plt.figure(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plots_dir = Path(output_path).parent / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
        corr_path = plots_dir / "corr_heatmap.png"; plt.savefig(corr_path, bbox_inches="tight"); plt.close()
        chart_b64 = encode_image_base64(corr_path)
        corr_html += f'<div><img src="{chart_b64}" alt="Correlation Heatmap"></div>'

    # ----------- Univariate Analysis -----------
    univariate_html = "<h2>Univariate Analysis</h2><table><tr>"
    col_count = 0
    for col in df.columns:
        plt.figure()
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
        else:
            df[col].value_counts().head(20).plot(kind="bar")
            plt.title(f"Top Categories in {col}")
        uni_path = plots_dir / f"uni_{sanitize_filename(col)}.png"
        plt.savefig(uni_path, bbox_inches="tight"); plt.close()

        # add chart cell
        univariate_html += f'<td style="padding:10px; text-align:center;"><h3>{col}</h3><img src="{encode_image_base64(uni_path)}" alt="Univariate {col}" style="max-width:500px;"></td>'
        col_count += 1

        # wrap to new row every 3 charts
        if col_count % 3 == 0:
            univariate_html += "</tr><tr>"

    univariate_html += "</tr></table>"

    # ----------- Multivariate Analysis -----------
    multivariate_html = "<h2>Multivariate Analysis</h2>"
    numeric_cols = plot_df.select_dtypes(include="number").columns
    if len(numeric_cols) > 1:
        sns.pairplot(plot_df[numeric_cols])
        pairplot_path = plots_dir / "pairplot.png"; plt.savefig(pairplot_path); plt.close()
        multivariate_html += f'<div><img src="{encode_image_base64(pairplot_path)}" alt="Pairplot"></div>'

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
    <footer style="position: fixed; bottom: 10px; left: 10px; font-size: 12px; color: gray;">
        DQA Report - Brock Frary - v1.0.2
    </footer>
</body>
</html>
"""
    template = Template(template_str)
    html_content = template.render(
        summary=summary_html,
        missing=missing_html,
        dup=dup_html,
        outliers=outlier_html,
        imbalance=imbalance_html,
        corr=corr_html,
        univariate=univariate_html,
        multivariate=multivariate_html
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return str(output_path)

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

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    return str(output_path)


# © 2025 Brock Frary. All rights reserved.