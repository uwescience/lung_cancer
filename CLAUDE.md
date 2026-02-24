# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-based clinical prediction system that uses Google's Gemini API to predict 2-year lung cancer survival outcomes from TCGA pathology reports (657 patients, LUAD cohort). The system supports zero-shot and few-shot prediction modes, both single-patient and batch.

## Environment Setup

```bash
source activate.sh   # activates venv (lun/) and sets PYTHONPATH to include src/
```

The Google Gemini API key is read at runtime from `/Users/jlheller/google_api_key_paid.txt`.

## Running Experiments

```bash
python scripts/run_experiments.py
```

Edit `scripts/run_experiments.py` before each run:
- Set `EXPERIMENT_PATH` to a new output file to avoid overwriting previous results
- Call `executeBatchMultishot(num_example=0)` for zero-shot, or `num_example=4` (or multiples of 4) for few-shot
- Call `zeroshotSingle()` for iterative single-patient mode

## Plotting results

```bash
python scripts/plot_results.py
```

## Running Tests

```bash
python -m unittest tests.test_bot
python -m unittest tests.test_multishot_maker
# or
nose2
```

Tests use `is_mock=True` on `Bot` to avoid real API calls.

## Architecture

### Data Flow

```
data/merged_data/processed_dataset.csv (657 patients)
  → Bot.__init__: assigns unique_id, selects columns
  → executeBatchMultishot / executeMultipleSingleZeroshot
  → (optional) MultishotMaker.buildExamples(): few-shot examples
  → prompts/{batch,zeroshot_single}/prompt1.py: prompt template
  → Gemini API (file upload for batch, chat for single)
  → CSV response parsed → merged with OS labels
  → experiments/<filename>.csv
```

### Implementation style
- Imports should be at the top of the module just after the module doc string

### Key Components

**`src/bot.py` — `Bot` class**
- `executeBatchMultishot(num_example)`: uploads patient data as a file to Gemini, processes unresponded patients in retry loops (up to 10 retries). Saves incrementally to experiment CSV.
- `executeMultipleSingleZeroshot(num_shot)`: iterates patients one at a time, new chat session per patient.
- `_executeGenerateContent()`: handles file upload, polling for processing, and response retrieval.
- `plotROC()` / `plotROCs()` / `plotPredictionRange()`: analysis helpers that read from experiment directories.
- `is_mock=True`: returns random predictions without API calls (for testing).
- Generation config: `temperature=0.0, top_p=1.0, top_k=1` (intended to be deterministic, but non-determinism has been observed in practice).

**`src/multishot_maker.py` — `MultishotMaker` class**
- `num_example` must be a positive multiple of 4: selects equal counts of survivors/non-survivors for each of Adenocarcinoma and Squamous Cell disease types.
- `buildExamples()` returns the example prompt string and the list of `unique_id`s used (so they are excluded from the prediction set).

**`src/constants.py`**
- All path constants (`PROJECT_DIR`, `DATA_DIR`, `EXPERIMENT_DIR`, etc.) and column name constants (`COL_PREDICTED`, `COL_ACTUAL`, `COL_UNIQUE_ID`, etc.) live here.

**`prompts/`**
- Each prompt file is a Python module with a `getPrompt()` function returning a string.
- `prompts/zeroshot_single/`: prompt string contains `%s` placeholder substituted with patient column data.
- `prompts/batch/`: prompt string for batch file-upload mode; few-shot examples are appended after.

### Experiment Output Format

Results CSVs contain: `unique_id`, `predicted` (float 0–1), `actual` (0/1 OS label).
Experiments are organized under `experiments/0shot/`, `experiments/4shot/`, `experiments/8shot/` for ROC comparison via `Bot.plotROCs()`.

### Known Issue

Non-deterministic outputs occur even with `temperature=0.0` — likely API-side randomness. Results differ between equivalent `(batch_size=1, num_batch=7)` vs `(batch_size=7, num_batch=1)` configurations.
