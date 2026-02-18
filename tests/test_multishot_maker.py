import src.constants as cn
from src.multishot_maker import (
    MultishotMaker, DISEASE_ADENOCARCINOMA, DISEASE_SQUAMOUS,
    COL_DISEASE_TYPE, COL_OS)

import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import unittest


IGNORE_TEST = False
TEST_DATA_PTH = os.path.join(cn.TEST_DIR, "test_data.csv")
TEST_DATA_DF = pd.read_csv(TEST_DATA_PTH)


class TestMultishotMaker(unittest.TestCase):

    def setUp(self):
        self.maker = MultishotMaker(TEST_DATA_DF)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsInstance(self.maker.data_df, pd.DataFrame)
        self.assertGreater(len(self.maker.data_df), 0)
        self.assertIsInstance(self.maker.examples, pd.DataFrame)
        self.assertEqual(len(self.maker.examples), 4)

    def testChooseExamples(self):
        if IGNORE_TEST:
            return
        examples = self.maker.examples
        # Should have 4 rows
        self.assertEqual(len(examples), 4)
        # Check disease types: first 2 adenocarcinoma, last 2 squamous
        self.assertEqual(examples.loc[0, COL_DISEASE_TYPE], DISEASE_ADENOCARCINOMA)
        self.assertEqual(examples.loc[1, COL_DISEASE_TYPE], DISEASE_ADENOCARCINOMA)
        self.assertEqual(examples.loc[2, COL_DISEASE_TYPE], DISEASE_SQUAMOUS)
        self.assertEqual(examples.loc[3, COL_DISEASE_TYPE], DISEASE_SQUAMOUS)
        # Check outcomes: survivor then non-survivor for each disease type
        self.assertEqual(examples.loc[0, COL_OS], 1)
        self.assertEqual(examples.loc[1, COL_OS], 0)
        self.assertEqual(examples.loc[2, COL_OS], 1)
        self.assertEqual(examples.loc[3, COL_OS], 0)
        # Each example should have a pathology report
        for idx in range(4):
            self.assertIsInstance(examples.loc[idx, cn.COL_PATHOLOGY_REPORT], str)
            self.assertGreater(len(examples.loc[idx, cn.COL_PATHOLOGY_REPORT]), 0)

    def testChooseExamplesRandomness(self):
        """Two MultishotMaker instances should (usually) pick different patients."""
        if IGNORE_TEST:
            return
        maker2 = MultishotMaker(TEST_DATA_DF)
        ids1 = self.maker.examples[cn.COL_SUBMITTER_ID].tolist()
        ids2 = maker2.examples[cn.COL_SUBMITTER_ID].tolist()
        # With random sampling, at least one should differ across many patients.
        # Run multiple trials to reduce flakiness.
        all_same = True
        for _ in range(5):
            maker_trial = MultishotMaker(TEST_DATA_DF)
            trial_ids = maker_trial.examples[cn.COL_SUBMITTER_ID].tolist()
            if trial_ids != ids1:
                all_same = False
                break
        self.assertFalse(all_same, "Expected different patients across instances")

    def testChooseExamplesMissingDiseaseType(self):
        """Should raise ValueError when a required disease type is missing."""
        if IGNORE_TEST:
            return
        # Create a dataframe with only adenocarcinoma patients
        adeno_df = self.maker.data_df[
            self.maker.data_df[COL_DISEASE_TYPE] == DISEASE_ADENOCARCINOMA].copy()
        with self.assertRaises(ValueError):
            MultishotMaker(adeno_df)

    def testBuildPrompt(self):
        if IGNORE_TEST:
            return
        prompt = self.maker.buildPrompt()
        self.assertIsInstance(prompt, str)
        # Should contain the instruction section
        self.assertIn("clinical oncologist", prompt)
        # Should contain all 4 examples
        self.assertIn("Example 1", prompt)
        self.assertIn("Example 2", prompt)
        self.assertIn("Example 3", prompt)
        self.assertIn("Example 4", prompt)
        # Should contain both outcome labels
        self.assertIn("survive", prompt)
        self.assertIn("not_survive", prompt)
        # Should contain the %s placeholder for the target patient
        self.assertIn("%s", prompt)
        # Should contain pathology reports from examples
        for idx in range(4):
            report = self.maker.examples.loc[idx, cn.COL_PATHOLOGY_REPORT]
            # Check that at least the first 50 chars of each report appear
            self.assertIn(report[:50], prompt)

    def testBuildPromptPlaceholder(self):
        """The prompt should be usable with string formatting."""
        if IGNORE_TEST:
            return
        prompt = self.maker.buildPrompt()
        patient_data = "pathology_report: Sample pathology text here."
        formatted = prompt % patient_data
        self.assertIn(patient_data, formatted)
        self.assertNotIn("%s", formatted)


if __name__ == '__main__':
    unittest.main()
