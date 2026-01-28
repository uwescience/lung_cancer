import os
from pathlib import Path

current_dir = os.path.abspath(os.getcwd())
PROJECT_DIR = Path(current_dir)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TEST_DIR = os.path.join(PROJECT_DIR, "tests")
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments")
EXPOSURE_PTH = os.path.join(DATA_DIR, "LUAD", "exposure.tsv")
CLINICAL_PTH = os.path.join(DATA_DIR, "LUAD", "clinical.tsv")
MERGED_DATA_PTH = os.path.join(DATA_DIR, "merged_data", "processed_dataset.csv")
# Column names
COL_PREDICTED = "predicted"
COL_ACTUAL = "actual"
COL_FILENAME = "filename"
COL_AUC = "AUC"
COL_PATHOLOGY_REPORT = "pathology_report"
COL_SUBMITTER_ID = "cases.submitter_id"