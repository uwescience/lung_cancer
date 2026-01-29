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
TEST_SMALL_DATA_PTH = os.path.join(cn.TEST_DIR, "test_small_data.csv")
TEST_EXPERIMENT_RESULT_PTH = os.path.join(cn.TEST_DIR, "experiment_result_dir",
        "oneshot_experiment1_results.csv")
TEST_EXPERIMENT_MULTIPLE_DIR = os.path.join(cn.TEST_DIR, "experiment_result_dir")
TEST_EXPERIMENT_FILENAME = "test_experiment.csv"
TEST_EXPERIMENT_PTH = os.path.join(cn.TEST_DIR, TEST_EXPERIMENT_FILENAME)
if not os.path.exists(TEST_DATA_PTH):
    MERGED_DATA_DF = pd.read_csv(cn.MERGED_DATA_PTH, sep=',')
    RANDOM_MERGED_DATA_DF = MERGED_DATA_DF.copy()
    for column in MERGED_DATA_DF.columns:
        RANDOM_MERGED_DATA_DF[column] = np.random.permutation(MERGED_DATA_DF[column])
    RANDOM_MERGED_DATA_DF.to_csv(TEST_DATA_PTH, index=False)

class TestBot(unittest.TestCase):

    def setUp(self):
        self.bot = Bot(diagnostic_pth=TEST_DATA_PTH,
            experiment_filename=TEST_EXPERIMENT_FILENAME,
            experiment_dir=cn.TEST_DIR,
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
            experiment_filename=TEST_EXPERIMENT_FILENAME,
            experiment_dir=cn.TEST_DIR,
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
        result_pth = os.path.join(cn.TEST_DIR, TEST_EXPERIMENT_FILENAME)
        if os.path.exists(result_pth):
            os.remove(result_pth)
        num_shots = 5
        results_df = self.bot.executeMultipleOneshot(num_shot=num_shots)
        self.assertEqual(len(results_df), num_shots)
        self.assertIn(cn.COL_PREDICTED, results_df.columns)
        self.assertIn(cn.COL_ACTUAL, results_df.columns)
        self.assertTrue(results_df[cn.COL_PREDICTED].apply(lambda x: isinstance(float(x), float)).all())
        self.assertTrue(results_df[cn.COL_ACTUAL].apply(lambda x: isinstance(float(x), float)).all())
        self.assertEqual(self.bot.oneshot_idx, num_shots)
        self.assertTrue(os.path.exists(TEST_EXPERIMENT_MULTIPLE_DIR))
        saved = pd.read_csv(result_pth)
        self.assertEqual(len(saved), num_shots)
        self.assertIn(cn.COL_PREDICTED, saved.columns)
        self.assertIn(cn.COL_ACTUAL, saved.columns)

        # Subsequent call should continue from the next index and append results
        additional_df = self.bot.executeMultipleOneshot(num_shot=3)
        self.assertEqual(len(additional_df), 3)
        self.assertEqual(self.bot.oneshot_idx, num_shots + 3)
        saved = pd.read_csv(TEST_EXPERIMENT_PTH)
        self.assertEqual(len(saved), num_shots + 3)

        # Large request should stop at dataset boundary
        remaining = self.bot.data_len - self.bot.oneshot_idx
        overflow_df = self.bot.executeMultipleOneshot(num_shot=remaining + 10)
        self.assertEqual(len(overflow_df), remaining)
        self.assertEqual(self.bot.oneshot_idx, self.bot.data_len)
        saved = pd.read_csv(TEST_EXPERIMENT_PTH)
        self.assertEqual(len(saved), num_shots + 3 + remaining)

        # A new bot with is_initialize=False should resume from existing file
        bot2 = Bot(diagnostic_pth=TEST_DATA_PTH,
            experiment_filename=TEST_EXPERIMENT_PTH,
            is_mock=True,
            is_initialize_experiment_file=False)
        self.assertEqual(bot2.oneshot_idx, self.bot.data_len)

    def testGetExperimentResults(self):
        if IGNORE_TEST:
            return
        result_dct = Bot.getExperimentResults(
            result_dir_name=TEST_EXPERIMENT_MULTIPLE_DIR,
            experiment_dir_pth=cn.TEST_DIR
        )
        self.assertIsInstance(result_dct, dict)
        self.assertIn("oneshot_experiment1_results.csv", result_dct)
        df = result_dct["oneshot_experiment1_results.csv"]
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn(cn.COL_PREDICTED, df.columns)
        self.assertIn(cn.COL_ACTUAL, df.columns)
    
    def testPlotPredictionRange(self):
        if IGNORE_TEST:
            return
        Bot.plotPredictionRange(
            result_dir_name=TEST_EXPERIMENT_MULTIPLE_DIR,
            experiment_dir_pth=cn.TEST_DIR, is_plot=IS_PLOT
        )

    def testPlotROCs(self):
        if IGNORE_TEST:
            return
        # Test with valid directory containing multiple CSV files
        Bot.plotROCs(TEST_EXPERIMENT_MULTIPLE_DIR, is_plot=IS_PLOT)

    def testExecuteMultipleOneshotInFile(self):
        if IGNORE_TEST:
            return
        if os.path.exists(TEST_EXPERIMENT_PTH):
            os.remove(TEST_EXPERIMENT_PTH)
        ##
        def test(data_path:str, is_mock: bool):
            expected_num = len(pd.read_csv(data_path))
            bot = Bot(
                diagnostic_pth=data_path,
                experiment_filename=TEST_EXPERIMENT_FILENAME,
                experiment_dir=cn.TEST_DIR,
                is_initialize_experiment_file=True,
                is_mock=is_mock)
            result_df = bot.executeMultipleOneshotInFile()
            if not (len(result_df) == expected_num):
                import pdb; pdb.set_trace()
            self.assertTrue(len(result_df) == expected_num)
            self.assertTrue(os.path.exists(TEST_EXPERIMENT_PTH))
        ##
        if IGNORE_TEST:
            print("small data, mock")
        test(data_path=TEST_SMALL_DATA_PTH, is_mock=True)
        if IGNORE_TEST:
            print("small data, not mock")
        test(data_path=TEST_SMALL_DATA_PTH, is_mock=False)
        #if IGNORE_TEST:
        #    print("large data, not mock")
        #test(data_path=cn.MERGED_DATA_PTH, is_mock=False)

    def test_ExecuteGenerateContent(self):
        if IGNORE_TEST:
            return
        # Load first entries from TEST_DATA_PTH
        num_entry = 3
        test_df = pd.read_csv(TEST_SMALL_DATA_PTH)
        input_df = test_df[self.bot.selected_columns].head(num_entry)
        ##
        def test(is_mock: bool):
            # Execute the method
            bot = Bot(diagnostic_pth=TEST_DATA_PTH,
                experiment_filename=TEST_EXPERIMENT_FILENAME,
                experiment_dir=cn.TEST_DIR,
                is_initialize_experiment_file=True,
                is_mock=is_mock
                )
            response_text, response = bot._executeGenerateContent(dataframe=input_df)
            # Verify response_text is a string
            self.assertIsInstance(response_text, str)
            # Verify response_text is not empty
            self.assertGreater(len(response_text), 0)
            # Verify response_text contains expected columns
            self.assertIn(cn.COL_SUBMITTER_ID, response_text)
            # Verify response_text contains predictions (comma-separated values)
            lines = response_text.strip().split('\n')
            # Should have header + num_entry data rows
            self.assertGreaterEqual(len(lines), 1 + num_entry)
            # Check that each line (except header) has comma-separated values
            for line in lines[1:1+num_entry]:  # Skip header, check first num_entry data lines
                parts = line.split(',')
                self.assertEqual(len(parts), 2)  # submitter_id, predicted
                # Second part should be convertible to float (probability)
                try:
                    prob = float(parts[1])
                    self.assertGreaterEqual(prob, 0.0)
                    self.assertLessEqual(prob, 1.0)
                except ValueError:
                    self.fail(f"Could not convert prediction to float: {parts[1]}")
        ##
        test(is_mock=True)
        test(is_mock=False)
        

if __name__ == '__main__':
    unittest.main()