# Developer: Brock Frary
# Title: file_picker.py
# Date: 09/25/2025
# Version: 0.3.0

"""
Purpose:
--------
Provides a Colab-friendly file picker utility for the Data Quality Analyzer (DQA).
- Defaults to /My Drive/DQA/live/data where input files are stored.
- Lists available files in that directory.
- Auto-selects if only one file exists.
- Otherwise, prompts user to choose from a numbered list.
"""

import os

def pick_file(base_dir="/content/drive/My Drive/DQA/live/data",
              filetypes=("*.csv", "*.xlsx", "*.xls", "*.json")):
    """
    Colab-friendly file picker.

    Parameters
    ----------
    base_dir : str
        Directory where input files are stored.
    filetypes : tuple
        Supported file extensions to filter.

    Returns
    -------
    str or None
        The full path of the selected file, or None if no selection is made.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Collect matching files
    files = [f for f in os.listdir(base_dir)
             if f.endswith(tuple(ext.replace("*", "") for ext in filetypes))]

    if not files:
        print(f"[FILE PICKER] No files found in {base_dir}")
        return None

    # Auto-select if only one file is found
    if len(files) == 1:
        selected = os.path.join(base_dir, files[0])
        print(f"[FILE PICKER] Only one file found. Auto-selected: {selected}")
        return selected

    # Otherwise, prompt user
    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    choice = input(f"Enter file number (1-{len(files)}) or press Enter to cancel: ")
    if not choice.strip():
        print("[FILE PICKER] No file selected.")
        return None

    try:
        idx = int(choice) - 1
        selected = os.path.join(base_dir, files[idx])
        print(f"[FILE PICKER] Selected file: {selected}")
        return selected
    except (ValueError, IndexError):
        print("[FILE PICKER] Invalid choice.")
        return None
