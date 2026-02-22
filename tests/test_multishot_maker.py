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
TEST_DATA_DF[cn.COL_UNIQUE_ID] = range(len(TEST_DATA_DF.index))
LEN_TEST_DATA = len(TEST_DATA_DF)


class TestMultishotMaker(unittest.TestCase):

    def setUp(self):
        self.num_examples = 8
        self.maker = MultishotMaker(TEST_DATA_DF, num_example=self.num_examples)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsInstance(self.maker.data_df, pd.DataFrame)
        self.assertGreater(len(self.maker.data_df), 0)
        self.assertIsInstance(self.maker.example_df, pd.DataFrame)
        self.assertEqual(len(self.maker.example_df), self.num_examples)

    def testChooseExamples(self):
        if IGNORE_TEST:
            return
        examples_df = self.maker.example_df
        per_group = self.num_examples // 4
        # Should have num_examples rows
        self.assertEqual(len(examples_df), self.num_examples)
        # Check disease types: first half adenocarcinoma, second half squamous
        for i in range(per_group):
            self.assertEqual(examples_df.loc[i, COL_DISEASE_TYPE], DISEASE_ADENOCARCINOMA)
            self.assertEqual(examples_df.loc[i, COL_OS], 1)
        for i in range(per_group, 2 * per_group):
            self.assertEqual(examples_df.loc[i, COL_DISEASE_TYPE], DISEASE_ADENOCARCINOMA)
            self.assertEqual(examples_df.loc[i, COL_OS], 0)
        for i in range(2 * per_group, 3 * per_group):
            self.assertEqual(examples_df.loc[i, COL_DISEASE_TYPE], DISEASE_SQUAMOUS)
            self.assertEqual(examples_df.loc[i, COL_OS], 1)
        for i in range(3 * per_group, 4 * per_group):
            self.assertEqual(examples_df.loc[i, COL_DISEASE_TYPE], DISEASE_SQUAMOUS)
            self.assertEqual(examples_df.loc[i, COL_OS], 0)
        # Each example should have a pathology report
        for idx in range(self.num_examples):
            self.assertIsInstance(examples_df.loc[idx, cn.COL_PATHOLOGY_REPORT], str)
            self.assertGreater(len(examples_df.loc[idx, cn.COL_PATHOLOGY_REPORT]), 0)   # type: ignore

    def testChooseExamplesRandomness(self):
        """Two MultishotMaker instances should (usually) pick different patients."""
        if IGNORE_TEST:
            return
        maker2 = MultishotMaker(TEST_DATA_DF)
        ids1 = self.maker.example_df[cn.COL_SUBMITTER_ID].tolist()
        ids2 = maker2.example_df[cn.COL_SUBMITTER_ID].tolist()
        # With random sampling, at least one should differ across many patients.
        # Run multiple trials to reduce flakiness.
        all_same = True
        for _ in range(5):
            maker_trial = MultishotMaker(TEST_DATA_DF)
            trial_ids = maker_trial.example_df[cn.COL_SUBMITTER_ID].tolist()
            if trial_ids != ids1:
                all_same = False
                break
        self.assertFalse(all_same, "Expected different patients across instances")

    def testChooseExamplesMissingDiseaseType(self):
        """Should raise ValueError when a required disease type is missing."""
        if IGNORE_TEST:
            return
        # Create a dataframe with only adenocarcinoma patients
        adeno_df = pd.DataFrame(self.maker.data_df[
            self.maker.data_df[COL_DISEASE_TYPE] == DISEASE_ADENOCARCINOMA].copy())
        with self.assertRaises(ValueError):
            MultishotMaker(adeno_df)

    def testBuildExamples(self):
        if IGNORE_TEST:
            return
        prompt, unique_ids = self.maker.buildExamples()
        self.assertIsInstance(prompt, str)
        # Should contain the instruction section
        self.assertIn("clinical oncologist", prompt)
        # Should contain all examples
        for i in range(1, self.num_examples + 1):
            self.assertIn(f"Example {i}", prompt)
        # Should contain both outcome labels
        self.assertIn("survive", prompt)
        self.assertIn("not_survive", prompt)
        # Should contain the %s placeholder for the target patient
        self.assertIn("%s", prompt)
        # Should contain pathology reports from examples
        for idx in range(self.num_examples):
            report = self.maker.example_df.loc[idx, cn.COL_PATHOLOGY_REPORT]
            # Check that at least the first 50 chars of each report appear
            self.assertIn(report[:50], prompt)  # type: ignore
        # Remaining DataFrame should exclude the examples
        self.assertEqual(len(unique_ids), self.num_examples)

    def testBuildPromptPlaceholder(self):
        """The prompt should be usable with string formatting."""
        if IGNORE_TEST:
            return
        prompt, _ = self.maker.buildExamples()
        patient_data = "pathology_report: Sample pathology text here."
        formatted = prompt % patient_data
        self.assertIn(patient_data, formatted)
        self.assertNotIn("%s", formatted)


if __name__ == '__main__':
    unittest.main()
