import src.constants as cn
from src.bot import Bot

import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import unittest


IGNORE_TEST = True
IS_PLOT = False
# Construct dummy test data
TEST_DATA_PTH = os.path.join(cn.TEST_DIR, "test_data.csv")
if not os.path.exists(TEST_DATA_PTH):
    MERGED_DATA_DF = pd.read_csv(cn.MERGED_DATA_PTH, sep=',')
    RANDOM_MERGED_DATA_DF = MERGED_DATA_DF.copy()
    for column in MERGED_DATA_DF.columns:
        RANDOM_MERGED_DATA_DF[column] = np.random.permutation(MERGED_DATA_DF[column])
    RANDOM_MERGED_DATA_DF.to_csv(TEST_DATA_PTH, index=False)

class TestBot(unittest.TestCase):

    def setUp(self):
        self.bot = Bot(path=TEST_DATA_PTH)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        bot = Bot(
            path=cn.MERGED_DATA_PTH,
            selected_columns=[
                "cases.submitter_id",
                "diagnoses.ajcc_pathologic_stage",
                "pathology_report",
                "diagnoses.residual_disease",
                "demographic.age_at_index"
            ],
            model="gemini-2.5-flash",
            key_path="/Users/jlheller/google_api_key_paid.txt"
        )
        self.assertEqual(bot.path, cn.MERGED_DATA_PTH)
        self.assertEqual(bot.model, "gemini-2.5-flash")
        self.assertEqual(
            bot.data_df.columns.tolist(),
            [
                "cases.submitter_id",
                "diagnoses.ajcc_pathologic_stage",
                "pathology_report",
                "diagnoses.residual_disease",
                "demographic.age_at_index"
            ]
        )

    def test_executeOneshot(self):
        #if IGNORE_TEST:
        #    return
        num_tests = 2
        [self.bot.executeOneshot() for _ in range(num_tests)]
        self.assertEqual(len(self.bot.oneshot_results), num_tests)
        trues = [isinstance(x, float) for x in self.bot.oneshot_results]
        self.assertTrue(all(trues))

if __name__ == '__main__':
    unittest.main()