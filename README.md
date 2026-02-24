# Lung Cancer Survival Prediction

An LLM-based clinical prediction system that uses Google's Gemini API to predict 2-year lung cancer survival outcomes from TCGA pathology reports. The dataset covers 657 patients from the LUAD (Lung Adenocarcinoma) cohort. The system supports zero-shot and few-shot prediction in both single-patient and batch modes.

## Setup

### Prerequisites

- Python 3.10+
- A Google Gemini API key saved to a file (default path: `/Users/jlheller/google_api_key_paid.txt`)

### Environment

```bash
source activate.sh
```

This activates the `lun/` virtual environment and sets `PYTHONPATH` to include `src/`.

## Project Structure

```text
lung_cancer/
├── data/
│   ├── LUAD/                        # Raw TCGA clinical and exposure TSVs
│   └── merged_data/
│       └── processed_dataset.csv    # 657-patient dataset used for experiments
├── experiments/
│   ├── 0shot/                       # Zero-shot experiment results
│   ├── 4shot/                       # 4-example few-shot results
│   └── 8shot/                       # 8-example few-shot results
├── prompts/
│   ├── batch/prompt1.py             # Prompt for batch (file-upload) mode
│   └── zeroshot_single/prompt1.py   # Prompt for single-patient mode
├── scripts/
│   └── run_experiments.py           # Main experiment runner
├── src/
│   ├── bot.py                       # Bot class: prediction, analysis, plotting
│   ├── constants.py                 # Paths and column name constants
│   └── multishot_maker.py           # Few-shot example selection and formatting
└── tests/
    ├── test_bot.py
    └── test_multishot_maker.py
```

## Running Experiments

Edit `scripts/run_experiments.py` to configure the run, then execute:

```bash
python scripts/run_experiments.py
```

### Configuration

Before each run, set `EXPERIMENT_PATH` to a new output file path to avoid overwriting previous results:

```python
EXPERIMENT_PATH = os.path.join(cn.EXPERIMENT_DIR, "my_experiment.csv")
```

### Zero-shot batch prediction (default)

Uploads all patient data as a file to Gemini and predicts in one pass, with up to 10 automatic retries for unresponded patients:

```python
executeBatchMultishot(num_example=0)
```

### Few-shot batch prediction

Include labeled examples in the prompt. `num_example` must be a positive multiple of 4 (equal survivors/non-survivors across both Adenocarcinoma and Squamous Cell subtypes):

```python
executeBatchMultishot(num_example=4)   # 4-shot
executeBatchMultishot(num_example=8)   # 8-shot
```

### Single-patient iterative mode

Processes one patient at a time with a fresh chat session per patient. Useful for debugging or smaller runs:

```python
zeroshotSingle()
```

Configure `batch_size` and `num_batch` in the script to control how many patients are processed per session and how many sessions are run.

## Output Format

Experiment results are saved as CSV files with columns:

| Column | Description |
| --- | --- |
| `unique_id` | Integer patient index (assigned at runtime) |
| `predicted` | Model output: float in [0, 1] (survival probability) |
| `actual` | Ground truth OS label: 1 = survived 2 years, 0 = did not |

Results are saved incrementally during the run, so partial results are preserved if a run is interrupted.

## Analysis

The `Bot` class provides plotting utilities that read from experiment directories.

### ROC curve for a single experiment

```python
import pandas as pd
from src.bot import Bot

df = pd.read_csv("experiments/0shot/my_experiment.csv")
Bot.plotROC(df)
```

### Compare ROC curves across experiment sets

```python
Bot.plotROCs(["0shot", "4shot", "8shot"])
```

Each directory may contain multiple replicate CSV files; the method also plots a median prediction curve across replicates.

### Prediction variability across replicates

```python
Bot.plotPredictionRange("0shot")
```

Plots the empirical CDF of per-patient prediction ranges across replicate runs, as a measure of output non-determinism.

## Running Tests

```bash
python -m unittest tests.test_bot
python -m unittest tests.test_multishot_maker
# or run all tests
nose2
```

Tests use `is_mock=True` on `Bot` to avoid real API calls.

## Known Behavior

**Non-determinism**: Results vary between runs even with `temperature=0.0, top_p=1.0, top_k=1`. This appears to be API-side randomness. Results also differ between equivalent configurations such as `(batch_size=1, num_batch=7)` vs. `(batch_size=7, num_batch=1)`. Running multiple replicates and using median predictions is recommended for more stable results.
