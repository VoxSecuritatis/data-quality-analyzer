# Changelog
All notable changes to the Data Quality Analyzer (DQA) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.7.2] - 2025-09-30
### Changed
- Updated **dqa_utils.py** to v2.7.2:
  - Added `export_pdf_report` using ReportLab for professional PDF exports.
  - Improved `export_html_report` with integrated schema export, duplicates, outlier detection, imbalance analysis, correlation, and NLP sections.
  - Enhanced `summarize_dataframe_html()` with best-practice guidelines and sampling recommendations.
  - Added automatic **plots folder cleanup** before regenerating charts.
  - Introduced **severity colorization** (`colorize_severity`) for better HTML readability.
  - Added **threshold parsing** (`get_cutoffs`) with support for percent or proportion formats.
  - Refined **filename sanitization** with `sanitize_filename()`.

### Fixed
- Alignment between runner and utilities (main runner was v2.7.0, now standardized at v2.7.2).

---

## [2.7.1] - 2025-09-29
### Added
- Improvements to **NLP reporting**:
  - Side-by-side comparison of **original vs. cleaned token frequencies**.
  - Highlighting differences (e.g., words dropped, stemmed, or normalized).
  - Explanatory bullets for analysts (context vs. modeling vocabulary).
  - Support for row wrapping in HTML layout (up to 4 tables per row for clarity).

### Changed
- Report formatting updates:
  - Added 2 blank lines at the bottom of generated HTML reports.
  - Improved HTML export structure with consistent section ordering.
- Export system now consistently timestamps outputs with dataset name.

---

## [2.7.0] - 2025-09-28
### Added
- **Main runner script (DQA_live-v2.7.0.py)**:
  - Clear screen at startup for cleaner console output.
  - Dynamic path resolution for `/src`, `/dst`, and `/data`.
  - Integrated **file picker** for dataset selection.
  - Added dataset summarization (`summarize_dataframe`) with:
    - Shape, head, tail, type info.
    - Auto-conversion of numeric-like objects.
    - Auto-conversion of datetime-like strings.
  - Added **missing value detection** with percentages per column.
  - Added **imputation recommendations** (Mean/Median for numeric, Mode for categorical).
  - Integrated **comparison mode toggle** for original vs. cleaned dataset.
  - Export of **self-contained HTML report** with timestamp + dataset name.

### Changed
- Console messages structured with **section headers** for better readability.
- Improved error handling for file selection and imports.

---

## [2.6] - 2025-09-26
### Added
- Roadmap expansion and **Gap Analysis Matrix**.
- Introduced **/live** and **/demo** run contexts:
  - **live** = production-ready with file picker for arbitrary datasets.
  - **demo** = static sample CSV for demonstration.
- Added support for **schema export** (`.expected.yaml`) to aid reproducibility and detect schema drift.

### Changed
- Updated `tree` structure and documentation to reflect new subdirectories.
- Clarified environment setup instructions and requirements.

---

## [2.5] - 2025-09-25
### Added
- Initial **Data Quality Analyzer (DQA) release**.
- Functional modules:
  - Dataset summary (shape, head, tail, stats).
  - Missing value analysis.
  - Duplicate detection.
  - Outlier detection.
  - Basic imbalance checks.
  - Correlation analysis.
- Report exports:
  - HTML summary reports.
  - CSV export of cleaned dataset.
- Documentation:
  - Added README.md.
  - Added roadmap notes.
- Baseline **plots folder** creation for charting outputs.

---

© 2025 Brock Frary. All rights reserved.
