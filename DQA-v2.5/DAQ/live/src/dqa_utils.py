# Developer: Brock Frary
# Title: dqa_utils.py
# Date: 09/25/2025
# Version: 0.4.0 (sampling parameterized, default 1000 rows; missing values table + chart included)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from jinja2 import Template
import base64
from io import StringIO
import re

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

# ----------- Helper: DataFrame Summary to HTML -----------
def summarize_dataframe_html(df: pd.DataFrame, n_rows: int = 5) -> str:
    html_parts = []
    html_parts.append(f"<h2>DataFrame Shape</h2><p>{df.shape[0]} rows × {df.shape[1]} columns</p>")
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    html_parts.append(f"<h2>Column Information</h2><pre>{info_str}</pre>")
    html_parts.append("<h2>First Rows (head)</h2>" + df.head(n_rows).to_html(classes="table table-striped", border=0))
    html_parts.append("<h2>Last Rows (tail)</h2>" + df.tail(n_rows).to_html(classes="table table-striped", border=0))
    try:
        html_parts.append("<h2>Descriptive Statistics (Numeric Columns)</h2>" +
                          df.describe(include=[float, int]).to_html(classes="table table-striped", border=0))
    except Exception as e:
        html_parts.append(f"<p><i>No numeric columns found. ({e})</i></p>")
    try:
        html_parts.append("<h2>Descriptive Statistics (Categorical Columns)</h2>" +
                          df.describe(include=[object]).to_html(classes="table table-striped", border=0))
    except Exception as e:
        html_parts.append(f"<p><i>No categorical columns found. ({e})</i></p>")
    return "\n".join(html_parts)

# ----------- Export HTML Report (Main Function) -----------
def export_html_report(df: pd.DataFrame, output_path="dst/dqa_report.html", sample_size: int = 1000):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary_html = summarize_dataframe_html(df)
    plot_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    # --- Missing Values Report (Counts + Percentages) ---
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing %": missing_pct.round(2)
    }).sort_values(by="Missing Count", ascending=False)
    missing_html = "<h2>Missing Values per Column</h2>" + missing_df.to_html(
        classes="table table-striped", border=0
    )

    # Add missing values bar chart
    if not missing_df.empty:
        plt.figure(figsize=(8, max(4, len(missing_df) * 0.4)))
        sns.barplot(
            x="Missing %", 
            y=missing_df.index, 
            hue=missing_df.index,   # map y axis to hue
            data=missing_df, 
            palette="Reds_r", 
            legend=False            # suppress legend
        )
        plt.title("Missing Values (%) by Column")
        plt.xlabel("Percentage Missing")
        plt.ylabel("Columns")

        # (fix path for missing values chart) =====
        chart_path = Path(output_path).parent / "missing_values.png"
        plt.savefig(chart_path, bbox_inches="tight"); plt.close()

        # Embed chart into HTML
        chart_b64 = encode_image_base64(chart_path)
        missing_html += f'<div><img src="{chart_b64}" alt="Missing Values Chart"></div>'

    # Example: Correlation Heatmap (sampled)
    numeric_cols = plot_df.select_dtypes(include="number").columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10,8))
        sns.heatmap(plot_df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")

    # ===== save correlation heatmap directly in dst =====
    corr_path = Path(output_path).parent / "corr_heatmap.png"
    plt.savefig(corr_path, bbox_inches="tight"); plt.close()

    # --- HTML Template ---
    template_str = """
<html>
<body>
    <h1>Data Quality Analyzer Report</h1>
    {{ summary | safe }}
    {{ missing | safe }}
    <footer style="position: fixed; bottom: 10px; left: 10px; font-size: 12px; color: gray;">
        DQA Demo - Brock Frary - v2.5 - 2025-09-25
    </footer>
</body>
</html>
"""
    template = Template(template_str)
    html_content = template.render(summary=summary_html, missing=missing_html)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return str(output_path)

# © 2025 Brock Frary. All rights reserved.