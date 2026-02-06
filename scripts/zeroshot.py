'''Does Zeroshot prediction using Gemini API.'''

"""
Notes
    1. When conducting a new experiment, change EXPERIMENT_PATH to a new file.
"""
import src.constants as cn
from src.bot import Bot

import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

EXPERIMENT_PATH = os.path.join(cn.EXPERIMENT_DIR, "zeroshot_experiment_test_results.csv")
REPORT_INTERVAL = 5


batch_size = 30
num_batch = 25
#batch_size = 3
#num_batch = 2
def zeroshotSingle():
    for count in range(num_batch):
        bot = Bot(experiment_filename=EXPERIMENT_PATH)
        bot.executeMultipleSingleZeroshot(num_shot=batch_size, prompt_file="prompt1.py")
        print(f"Completed {(count + 1) * batch_size} out of {batch_size * num_batch} zero-shot predictions.")
    print(f"Results saved to {EXPERIMENT_PATH}.")

def zeroshotBatch():
    print(f"Executing zero-shot predictions sequentially, saving results to {EXPERIMENT_PATH}...")
    bot = Bot(
        experiment_filename=EXPERIMENT_PATH)
    result_df = bot.executeBatchZeroshot(prompt_file="prompt1.py")
    print(f"Completed {len(result_df)} zero-shot predictions.")
    print(f"Results saved to {EXPERIMENT_PATH}.")
        

if __name__ == '__main__':
    zeroshotBatch()