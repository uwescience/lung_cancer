import os
from pathlib import Path

current_dir = os.path.abspath(os.getcwd())
PROJECT_DIR = Path(current_dir)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TEST_DIR = os.path.join(PROJECT_DIR, "tests")
EXPOSURE_PTH = os.path.join(DATA_DIR, "LUAD", "exposure.tsv")
CLINICAL_PTH = os.path.join(DATA_DIR, "LUAD", "clinical.tsv")
MERGED_DATA_PTH = os.path.join(DATA_DIR, "merged_data", "processed_dataset.csv")