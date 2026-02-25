# Test Creator Agent Memory

See `patterns.md` for detailed notes.

## Key Facts (quick reference)

- Test files: `tests/test_bot.py`, `tests/test_multishot_maker.py`
- Run tests: `source activate.sh && python -m unittest tests.test_bot`
- `Bot` setUp pattern: `Bot(diagnostic_pth=TEST_DATA_PTH, experiment_filename=TEST_EXPERIMENT_FILENAME, experiment_dir=cn.TEST_DIR, is_initialize_experiment_file=True, is_mock=True)`
- Default `selected_data_df` columns (with default `selected_columns`): `["pathology_report", "unique_id"]` — `cases.submitter_id` is stripped and `unique_id` appended
- All constants (paths, column names) live in `src/constants.py`; import as `cn`
- `cn.COL_PATHOLOGY_REPORT = "pathology_report"`, `cn.COL_UNIQUE_ID = "unique_id"`
- IGNORE_TEST and IS_TEST are module-level booleans; new/revised tests guard with `if IGNORE_TEST: return` as the first two lines (commented out for new/revised tests)
- Naming convention confirmed: `test_<method_name>_<scenario>` (snake_case), e.g. `test_getData_returns_selected_dataframe`
- Older tests use camelCase `testXxx` — rename to snake_case when revising
- `getData()` returns `self.selected_data_df` directly (same object reference, not a copy); assertIs is valid
- `data_len` is set from `len(self.full_data_df.index)` before column selection; row count of `selected_data_df` always equals `data_len`
