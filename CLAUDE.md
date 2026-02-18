# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A lung cancer survival prediction project using the Google Gemini API to analyze pathology reports and clinical data (LUAD dataset) to predict 2-year survival outcomes via zero-shot prompting.

## Environment Setup

```bash
source activate.sh  # Activates virtualenv (lun/) and sets PYTHONPATH to include src/
```

`activate.sh` adds `src/` to `PYTHONPATH`, which is required for `import src.constants as cn` to work. **All scripts must be run from the project root.**

`src/constants.py` uses `os.getcwd()` to compute paths, so running from the wrong directory will break all file paths.

## Running Experiments

```bash
python ./scripts/zeroshot.py       # Runs executeBatchZeroshot() by default
```

To run single-patient sequential predictions instead, call `zeroshotSingle()` from within the script (swap the `if __name__ == '__main__'` call).

## Running Tests

```bash
python ./tests/test_bot.py         # Runs all unittest tests
```

Set `IGNORE_TEST = True` in `test_bot.py` to skip slow/API-dependent tests. Use `is_mock=True` in `Bot()` to test without API calls.

## Architecture

### Core: `src/bot.py` — `Bot` class

The central orchestrator. Key constructor parameters:
- `diagnostic_pth`: CSV data file (default: `data/merged_data/processed_dataset.csv`)
- `selected_columns`: columns fed to the prompt (default: `["cases.submitter_id", "pathology_report"]`)
- `experiment_filename`: output CSV name; auto-resumes from existing file by checking its length to set `zeroshot_idx`
- `is_initialize_experiment_file`: if `True`, deletes existing experiment file before starting
- `is_mock`: returns `np.random.uniform(0,1)` instead of making API calls (for testing)
- `is_randomized`: randomly permutes each selected column (used in tests to prevent data leakage)

Two prediction modes:
- **Single** (`executeSingleZeroshot` / `executeMultipleSingleZeroshot`): one API chat per patient, sequential, appends to CSV
- **Batch** (`executeBatchZeroshot`): uploads all patient data as a CSV file to Gemini, retries up to 10 times for unprocessed patients, uses `_executeGenerateContent` which writes to `data/local_context.csv` then uploads

### Prompts: `prompts/`

Organized into subdirectories by mode:
- `prompts/zeroshot_single/` — single-patient prompts (string with `%s` placeholder for patient data)
- `prompts/zeroshot_batch/` — batch prompts (references the uploaded file)

Each prompt file is a Python module with a `getPrompt()` function. Loaded dynamically via `__import__` in `Bot._getPrompt(directory, prompt_file)`.

### Paths & Constants: `src/constants.py`

All file paths centralized here. Update `MERGED_DATA_PTH` or `EXPERIMENT_DIR` here when changing data locations.

### Data

- `data/merged_data/processed_dataset.csv`: main dataset; key columns are `cases.submitter_id`, `pathology_report`, `OS` (binary label: 1=survived 2+ years, 0=died)
- `data/local_context.csv`: temporary file written during batch processing; not committed

### Experiments: `experiments/`

Results organized in subdirectories (e.g., `experiments/simple/`, `experiments/batch/`). Each CSV has: `cases.submitter_id`, selected columns, `predicted` (float 0–1), `actual` (OS label).

Use `Bot.getExperimentResults(result_dir_name)` to load a directory of CSVs into `Dict[str, pd.DataFrame]`. Use `Bot.plotROCs(result_dir_names)` to overlay ROC curves; it also computes a "Median Prediction" across runs.

### Gemini API

- API key read from file (default: `/Users/jlheller/google_api_key_paid.txt`), stored to `GEMINI_API_KEY` env var
- Model default: `gemini-2.5-flash`
- Single mode: `client.chats.create()` + `chat.send_message(prompt)`
- Batch mode: `client.files.upload()` + `client.models.generate_content(model, contents=[prompt, file])`

## Conventions

- Method names are camelCase (`executeSingleZeroshot`, `plotROCs`)
- Lists end in "s" (`selected_columns`, `result_dcts`); integer counts do not
- `IGNORE_TEST = False` at top of `test_bot.py` controls test skipping globally
- `IS_PLOT = False` in tests suppresses matplotlib display during test runs
- DataFrames end in "_df". Global names of dataframes (in capitals) end in "_DF"
