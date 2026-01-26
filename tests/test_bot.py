import src.constants as cn
from src.bot import Bot

import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import unittest


IGNORE_TEST = False
IS_PLOT = False
# Construct dummy test data
TEST_DATA_PTH = os.path.join(cn.TEST_DIR, "test_data.csv")
TEST_EXPERIMENT_PATH = os.path.join(cn.TEST_DIR, "test_experiment.csv")
if not os.path.exists(TEST_DATA_PTH):
    MERGED_DATA_DF = pd.read_csv(cn.MERGED_DATA_PTH, sep=',')
    RANDOM_MERGED_DATA_DF = MERGED_DATA_DF.copy()
    for column in MERGED_DATA_DF.columns:
        RANDOM_MERGED_DATA_DF[column] = np.random.permutation(MERGED_DATA_DF[column])
    RANDOM_MERGED_DATA_DF.to_csv(TEST_DATA_PTH, index=False)

class TestBot(unittest.TestCase):

    def setUp(self):
        self.bot = Bot(diagnostic_pth=TEST_DATA_PTH,
            experiment_filename=TEST_EXPERIMENT_PATH,
            is_initialize_experiment_file=True,
            is_mock=True)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        bot = Bot(
            diagnostic_pth=cn.MERGED_DATA_PTH,
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
            bot.selected_data_df.columns.tolist(),
            [
                "cases.submitter_id",
                "diagnoses.ajcc_pathologic_stage",
                "pathology_report",
                "diagnoses.residual_disease",
                "demographic.age_at_index"
            ]
        )

    def testExecuteOneshot(self):
        if IGNORE_TEST:
            return
        bot = Bot(diagnostic_pth=TEST_DATA_PTH,
            experiment_filename=TEST_EXPERIMENT_PATH,
            is_mock=False)
        if True:
            num_test = 2
            results = [bot.executeOneshot(data_idx=n) for n in range(num_test)]
            result_dct = {key: [d[key] for d in results] for key in results[0]}
            trues = [isinstance(x, float) for x in result_dct[cn.COL_PREDICTED]]
            self.assertTrue(all(trues))
        #
        num_test = 1000
        with self.assertRaises(IndexError):
            _ = [self.bot.executeOneshot(data_idx=n) for n in range(num_test)]
        #
        num_test = 600
        results = [self.bot.executeOneshot(data_idx=n) for n in range(num_test)]
        result_dct = {key: [d[key] for d in results] for key in results[0]}
        trues = [isinstance(x, float) for x in result_dct[cn.COL_PREDICTED]]
        self.assertTrue(all(trues))
        trues = [isinstance(float(x), float) for x in result_dct[cn.COL_ACTUAL]]
        self.assertTrue(all(trues))
        self.assertLessEqual(np.abs(np.mean(result_dct[cn.COL_PREDICTED])-0.5), 0.1)

    def testExecuteMultipleOneshot(self):
        if IGNORE_TEST:
            return
        if os.path.exists(TEST_EXPERIMENT_PATH):
            os.remove(TEST_EXPERIMENT_PATH)
        num_shots = 5
        results_df = self.bot.executeMultipleOneshot(num_shot=num_shots)
        self.assertEqual(len(results_df), num_shots)
        self.assertIn(cn.COL_PREDICTED, results_df.columns)
        self.assertIn(cn.COL_ACTUAL, results_df.columns)
        self.assertTrue(results_df[cn.COL_PREDICTED].apply(lambda x: isinstance(float(x), float)).all())
        self.assertTrue(results_df[cn.COL_ACTUAL].apply(lambda x: isinstance(float(x), float)).all())
        self.assertEqual(self.bot.oneshot_idx, num_shots)
        self.assertTrue(os.path.exists(TEST_EXPERIMENT_PATH))
        saved = pd.read_csv(TEST_EXPERIMENT_PATH)
        self.assertEqual(len(saved), num_shots)
        self.assertIn(cn.COL_PREDICTED, saved.columns)
        self.assertIn(cn.COL_ACTUAL, saved.columns)

        # Subsequent call should continue from the next index and append results
        additional_df = self.bot.executeMultipleOneshot(num_shot=3)
        self.assertEqual(len(additional_df), 3)
        self.assertEqual(self.bot.oneshot_idx, num_shots + 3)
        saved = pd.read_csv(TEST_EXPERIMENT_PATH)
        self.assertEqual(len(saved), num_shots + 3)

        # Large request should stop at dataset boundary
        remaining = self.bot.data_len - self.bot.oneshot_idx
        overflow_df = self.bot.executeMultipleOneshot(num_shot=remaining + 10)
        self.assertEqual(len(overflow_df), remaining)
        self.assertEqual(self.bot.oneshot_idx, self.bot.data_len)
        saved = pd.read_csv(TEST_EXPERIMENT_PATH)
        self.assertEqual(len(saved), num_shots + 3 + remaining)

        # A new bot with is_initialize=False should resume from existing file
        bot2 = Bot(diagnostic_pth=TEST_DATA_PTH,
            experiment_filename=TEST_EXPERIMENT_PATH,
            is_mock=True,
            is_initialize_experiment_file=False)
        self.assertEqual(bot2.oneshot_idx, self.bot.data_len)


if __name__ == '__main__':
    unittest.main()