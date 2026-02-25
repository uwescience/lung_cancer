# Test Cleanup Agent Memory

## Guard Line vs. Disabled Code Distinction

In test_bot.py, lines like:
```python
#if IGNORE_TEST:
#    print("large data, not mock")
#test(data_path=cn.MERGED_DATA_PTH, is_mock=False)
```
are intentionally disabled test scenarios (not guards). Do NOT uncomment these.

True guards follow the pattern:
```python
# if IGNORE_TEST:
#     return
```
or similar — a conditional that would cause the test method to return/skip early. These should be uncommented.

## Project Test Files

- `/Users/jlheller/home/Technical/repos/lung_cancer/tests/test_bot.py`
- `/Users/jlheller/home/Technical/repos/lung_cancer/tests/test_multishot_maker.py`

## Typical Commit-Ready State

- `IGNORE_TEST = False` (both files)
- `IS_PLOT = False` (test_bot.py only; test_multishot_maker.py has no IS_PLOT)
- All `# if IGNORE_TEST:` / `#     return` guard pairs uncommented
