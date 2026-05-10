"""
Per-file 80/20 split script for ECG SR vs ST classification.
No file picking uses files that were previously picked.
"""

import csv
from pathlib import Path

import numpy as np

# hardcoded paths
ROOT_FOLDER = Path(r"C:\Users\Laura\ECG_10000")

# Input: raw files, folders train and test 
RAW_INPUT_ROOT = ROOT_FOLDER / "Dataset_RawFiles_Lead2_ST_secondtry"

# Output root
OUT_ROOT = ROOT_FOLDER / "Dataset_PerFile_8020_Lead2_ST"

TRAIN_RATIO = 0.8

# ECG sampling parameters 
FS = 500
TIMESTAMP_STEP_MS = 1000 // FS   # 2 ms at 500 Hz

MIN_SECONDS = 2


# help functions
def read_raw_csv(path: Path) -> np.ndarray:
    """Read a raw CSV (timestamp, leadII) and return the leadII column."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        signal = [float(row[1]) for row in reader if len(row) >= 2]
    return np.array(signal, dtype=np.float32)


def write_signal_csv(path: Path, signal: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "leadII"])
        for i, v in enumerate(signal):
            w.writerow([i * TIMESTAMP_STEP_MS, f"{float(v):.6f}"])


def list_all_input_files(input_root: Path) -> list:
    """Return all CSV files from training/ and testing/ subfolders."""
    files = []
    for split in ("training", "testing"):
        sub = input_root / split
        if sub.exists():
            files.extend(sorted(sub.glob("*.csv")))
    files.extend(sorted(input_root.glob("*.csv")))
    return files


def main():
    if not RAW_INPUT_ROOT.exists():
        print(f"ERROR: input folder not found: {RAW_INPUT_ROOT}")
        return

    files = list_all_input_files(RAW_INPUT_ROOT)
    print(f"Input root: {RAW_INPUT_ROOT}")
    print(f"Found {len(files)} CSV files to split.\n")

    train_out = OUT_ROOT / "training"
    test_out  = OUT_ROOT / "testing"
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skipped_short = 0
    n_skipped_error = 0

    for fp in files:
        try:
            signal = read_raw_csv(fp)
        except (ValueError, IndexError) as e:
            print(f"  Skipped (read error): {fp.name} ({e})")
            n_skipped_error += 1
            continue

        if len(signal) < FS * MIN_SECONDS:
            n_skipped_short += 1
            continue

        # Per-file split: first 80% -> train, last 20% -> test
        split_idx = int(len(signal) * TRAIN_RATIO)
        train_part = signal[:split_idx]
        test_part  = signal[split_idx:]

        # Both parts share the original filename, just in different folders
        write_signal_csv(train_out / fp.name, train_part)
        write_signal_csv(test_out  / fp.name, test_part)
        n_ok += 1

    print(f"\nDone.")
    print(f"  Files split successfully : {n_ok}")
    print(f"  Skipped (too short)      : {n_skipped_short}")
    print(f"  Skipped (read error)     : {n_skipped_error}")
    print(f"\nOutput:")
    print(f"  {train_out}")
    print(f"  {test_out}")


if __name__ == "__main__":
    main()
