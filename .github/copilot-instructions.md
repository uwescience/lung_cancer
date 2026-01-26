# Copilot Instructions for Lung Cancer Analysis Project

## Project Overview
This is a **lung cancer survival prediction project** using the Google Gemini API to analyze pathology reports and clinical data. The core workflow uses one-shot prompting with clinical context to predict 2-year survival outcomes. Data comes from LUAD (Lung Adenocarcinoma) datasets with pathology reports and survival labels.

## Architecture & Data Flow
- **Data Layer**: CSV/TSV files in `data/` (clinical, exposure data, merged processed dataset)
- **Core Logic** (`src/bot.py`): `Bot` class orchestrates:
  - CSV data loading with pandas
  - Gemini API client initialization (requires API key from environment)
  - One-shot prediction execution per patient row
  - Results collection and CSV export to `experiments/`
- **Utilities** (`src/constants.py`): Path management for data directories
- **Scripts** (`scripts/oneshot.py`): Batch runner executing 600+ predictions with interval reporting
- **Testing** (`tests/test_bot.py`): Unit tests with mock data generation and `is_mock` flag support

## Critical Patterns & Conventions
- **Naming**: Lists end in an "s" (e.g., `selected_columns`, `oneshot_results`). ints that are counts do not.
- **Path Management**: All file paths centralized in `src/constants.py` for easy updates
- **Stateless Execution**: Each call to `executeOneshot(data_idx)` is independent; no internal state retained between calls
- **Prompting**: One-shot prompt template defined in `ONESHOT_PROMPT` with `%s` placeholders for dynamic insertion of clinical text
- Method names are camelCase (e.g., `executeOneshot`, `calculateAUC`)

### Data Handling
- Merged dataset in `data/merged_data/processed_dataset.csv` contains columns:
  - `cases.submitter_id`: Patient identifier
  - `pathology_report`: Clinical text (used in prompts)
  - `OS`: 2-year survival binary label (1=survived, 0=died)
- Tests use randomized permutations of production data (see `test_bot.py` line 16-19)

### Gemini API Integration
- One-shot prompt template in `ONESHOT_PROMPT` (lines 13-21 of `bot.py`) instructs model as "clinical oncologist" 
- Returns float probability (0-1) of 2-year survival
- API key stored in file path (default: `/Users/jlheller/google_api_key_paid.txt`) and loaded to `GEMINI_API_KEY` environment variable
- **Active method**: `executeOneshot(data_idx)` creates new chat per request (deprecated `deprecatedexecuteOneshot()` batch mode exists)

### Experiment Workflow
- Results saved to `experiments/{random_7_digit_id}.csv` (auto-generated if filename not provided)
- Each row contains: selected columns + `predicted` (model output) + `actual` (ground truth OS)
- AUC calculated via `sklearn.roc_auc_score` (see `calculateAUC()`)

### Testing & Mocking
- Use `is_mock=True` in `Bot()` constructor to return random predictions (no API calls)
- Tests construct dummy data if `test_data.csv` doesn't exist
- `IGNORE_TEST` flag (line 10 of `test_bot.py`) controls test execution

## Developer Workflows

### Running Experiments
```bash
# Batch prediction (600 samples with progress every 5)
python ./scripts/oneshot.py

# Single prediction via test
python ./tests/test_bot.py
```

### Setting Up Environment
```bash
source activate.sh  # Activates Python virtual environment
```

### Key Dependencies
- `google-genai`: Gemini API client
- `pandas`, `numpy`: Data manipulation
- `sklearn`: AUC/ROC metrics
- `jupyter`, `jupyterlab`: Notebooks for exploration
- ML libraries: `tensorflow`, `torch`, `keras`, `scipy`

## Code Organization Tips
1. **Constants**: All path management in `src/constants.py` - update `MERGED_DATA_PTH` and `EXPERIMENT_DIR` there
2. **Bot state**: `executeOneshot()` is stateless per-row; `oneshot_results` list accumulates predictions across calls
3. **Prompts**: `ONESHOT_PROMPT` template uses `%s` string formatting - modify clinical instructions here
4. **Error handling**: Raises `ValueError` if API returns `None` or missing columns detected during initialization

## Common Tasks for AI Agents
- **Add metrics**: Modify `calculateAUC()` or create new methods using `self.oneshot_results` and ground truth labels
- **Adjust prompts**: Update `ONESHOT_PROMPT` clinical context (lines 13-21)
- **Change data columns**: Edit `selected_columns` parameter in `Bot()` calls or `testConstructor()` example
- **Debug predictions**: Check experiment CSV in `experiments/` - compare `predicted` vs `actual` columns
- **Scaling**: Batch size managed by `scripts/oneshot.py` loop and `REPORT_INTERVAL` for progress tracking
