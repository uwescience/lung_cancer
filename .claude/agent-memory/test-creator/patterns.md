# Test Patterns and Conventions

## Naming
- Method name format: `test<MethodName>` (camelCase after "test"), e.g., `testGetData`, `testConstructor`
- One exception observed: `test_ExecuteGenerateContent` — generally prefer camelCase

## Guard Pattern
Every test method starts with:
```python
if IGNORE_TEST:
    return
```
IGNORE_TEST is False at module level (all tests run by default). When writing a new test, comment out these two lines so the new test is active even if the reviewer sets IGNORE_TEST = True temporarily.

## Bot Initialization in Tests
- Always pass `is_mock=True` to avoid real API calls
- setUp uses `TEST_DATA_PTH` (randomly permuted copy of production CSV, created once)
- `is_initialize_experiment_file=True` ensures a fresh experiment file per test

## selected_data_df Column Logic (from Bot.__init__)
- `cases.submitter_id` is always removed from selected_columns
- `unique_id` (`cn.COL_UNIQUE_ID`) is always appended
- Default call yields columns: `["pathology_report", "unique_id"]`

## Assertion Style
- Use descriptive failure messages as the third arg to assert methods
- `assertIsInstance`, `assertFalse(df.empty)`, `assertIn(col, df.columns)` are the standard DataFrame checks

## Constants to use (from src/constants.py)
- `cn.COL_UNIQUE_ID` = `"unique_id"`
- `cn.COL_PREDICTED` = `"predicted"`
- `cn.COL_ACTUAL` = `"actual"`
- `cn.COL_PATHOLOGY_REPORT` = `"pathology_report"`
- `cn.COL_SUBMITTER_ID` = `"cases.submitter_id"`
- `cn.TEST_DIR`, `cn.EXPERIMENT_DIR`, `cn.MERGED_DATA_PTH`
