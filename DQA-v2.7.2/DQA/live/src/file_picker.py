# file_picker.py
# Version: 2.7.2
# Developer: Brock Frary
# Date: 2025-09-30

"""
Purpose:
--------
Provides a Colab and local-friendly file picker utility for the Data Quality Analyzer (DQA).

Features:
- Detects execution environment (Google Colab vs Windows local).
- Prefers DQA_DATA_PATH environment variable if defined.
- Defaults to environment-appropriate /data directory.
- Lists available files in that directory.
- Auto-selects if only one file exists.
- Otherwise, prompts user to choose from a numbered list.
"""

import os
import sys

# ----------- Detect environment and set default base_dir -----------
def detect_base_dir():
    """
    Detects the most appropriate base directory for datasets depending on the environment.
    Priority order:
    1. Environment variable DQA_DATA_PATH (if defined and valid)
    2. Google Colab default: '/content/drive/My Drive/DQA/live/data'
    3. Windows local dev: './data' if it exists, else current working directory
    4. Fallback: current working directory
    """
    # ----- Check explicit environment variable first -----
    env_path = os.getenv("DQA_DATA_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # ----- If running in Google Colab -----
    if "google.colab" in sys.modules:
        return "/content/drive/My Drive/DQA/live/data"

    # ----- If running on Windows (local dev) -----
    elif os.name == "nt":
        local_path = os.path.join(os.getcwd(), "data")
        if os.path.exists(local_path):
            return local_path
        return os.getcwd()

    # ----- Fallback: current working directory -----
    else:
        return os.getcwd()

# ----------- File picker function -----------
def pick_file(base_dir=None,
              filetypes=("*.csv", "*.xlsx", "*.xls", "*.json")):
    """
    Colab and local-friendly file picker.

    Parameters
    ----------
    base_dir : str or None
        Directory where input files are stored. If None, detect_base_dir() is used.
    filetypes : tuple
        Supported file extensions to filter.

    Returns
    -------
    str or None
        The full path of the selected file, or None if no selection is made.
    """
    # ----- Ensure base_dir is set appropriately -----
    if base_dir is None:
        base_dir = detect_base_dir()

    if not os.path.exists(base_dir):
        print(f"[FILE PICKER] Data directory not found: {base_dir}")
        return None

    # ----- Collect matching files -----
    files = [f for f in os.listdir(base_dir)
             if f.endswith(tuple(ext.replace("*", "") for ext in filetypes))]

    if not files:
        print(f"[FILE PICKER] No files found in {base_dir}")
        return None

    # ----- Auto-select if only one file -----
    if len(files) == 1:
        selected = os.path.join(base_dir, files[0])
        print(f"[FILE PICKER] Only one file found. Auto-selected: {selected}")
        return selected

    # ----- Otherwise, prompt user -----
    print("\nAvailable files:")
    print("  0. Exit (to add CSV source dataset)")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    while True:
        choice = input(f"Enter file number (0-{len(files)}) or press Enter to cancel: ").strip()
        if not choice:
            print("[FILE PICKER] No file selected.")
            return None
        try:
            idx = int(choice)
            if idx == 0:
                return None
            elif 1 <= idx <= len(files):
                selected = os.path.join(base_dir, files[idx - 1])
                print(f"[FILE PICKER] Selected file: {selected}")
                return selected
            else:
                print(f"[FILE PICKER] Invalid choice. Please enter a number between 0 and {len(files)}.")
        except ValueError:
            print("[FILE PICKER] Invalid input. Please enter a valid number.")

# © 2025 Brock Frary. All rights reserved.