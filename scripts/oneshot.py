'''Does One-Shot prediction using Gemini API.'''

"""
Notes
    1. When conducting a new experiment, change EXPERIMENT_FILENAME to a new file.
"""
import src.constants as cn
from src.bot import Bot

import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

EXPERIMENT_FILENAME = os.path.join(cn.EXPERIMENT_DIR, "oneshot_experiment100_results.csv")
REPORT_INTERVAL = 5


batch_size = 30
num_batch = 25
#batch_size = 3
#num_batch = 2
def oneshotSequential():
    for count in range(num_batch):
        bot = Bot(experiment_filename=EXPERIMENT_FILENAME)
        bot.executeMultipleOneshot(num_shot=batch_size)
        print(f"Completed {(count + 1) * batch_size} out of {batch_size * num_batch} one-shot predictions.")

def oneshotInFile():
    expected_num = 658
    bot = Bot(
        experiment_filename=EXPERIMENT_FILENAME)
    result_df = bot.executeMultipleOneshotInFile()
    if not (len(result_df) == expected_num):
        import pdb; pdb.set_trace()
        

if __name__ == '__main__':
    oneshotInFile()