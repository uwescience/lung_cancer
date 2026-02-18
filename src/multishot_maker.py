'''Builds few-shot prompts using example patients from the dataset.'''

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import src.constants as cn


DISEASE_ADENOCARCINOMA = "Adenomas and Adenocarcinomas"
DISEASE_SQUAMOUS = "Squamous Cell Neoplasms"
COL_DISEASE_TYPE = "cases.disease_type"
COL_OS = "OS"


class MultishotMaker(object):
    '''Selects example patients and builds few-shot prompts.'''

    def __init__(self, data_df: pd.DataFrame) -> None:
        """
        Args:
            data_df (pd.DataFrame): DataFrame of patients.
        """
        self.data_df = data_df
        self.examples = self._chooseExamples()

    def _chooseExamples(self) -> pd.DataFrame:
        """Chooses 4 example patients: one survivor and one non-survivor
        for each of adenocarcinoma and squamous cell cancer.

        Returns:
            pd.DataFrame: 4 rows with columns from the original dataset.
        """
        selections = []
        for disease_type in [DISEASE_ADENOCARCINOMA, DISEASE_SQUAMOUS]:
            disease_df = self.data_df[self.data_df[COL_DISEASE_TYPE] == disease_type]
            for outcome in [1, 0]:
                candidates = disease_df[disease_df[COL_OS] == outcome]
                if len(candidates) == 0:
                    raise ValueError(
                        f"No patients found for {disease_type} with OS={outcome}")
                row = candidates.sample(n=1, random_state=None)
                selections.append(row)
        return pd.concat(selections, ignore_index=True)

    def buildPrompt(self) -> str:
        """Builds a few-shot prompt using the 4 example patients.
        The prompt uses only the pathology report as the example data.

        Returns:
            str: Prompt with examples and a %%s placeholder for the target patient data.
        """
        outcome_labels = {1: "survive", 0: "not_survive"}
        examples_text = ""
        for idx, row in self.examples.iterrows():
            idx = int(idx)  # type: ignore
            label = outcome_labels[row[COL_OS]]
            report = row[cn.COL_PATHOLOGY_REPORT].replace("%", "%%")
            examples_text += (
                f"Example {idx + 1}\n"
                f"Pathology report:\n"
                f"{report}\n"
                f"Answer:\n"
                f"{label}\n\n"
            )
        prompt = (
            "Instruction:\n"
            "You are a clinical oncologist with expertise in cancer prognosis.\n\n"
            "Task:\n"
            "Based on the pathology report, predict the probability that the "
            "patient survived beyond 2 years from the date of diagnosis.\n\n"
            "Here are some example cases and their outcomes:\n\n"
            f"{examples_text}"
            "Now predict for the following case:\n"
            "%s\n"
            "Output format (no explanation):\n"
            "Return only a single number between 0 and 1 representing the "
            "probability of surviving beyond 2 years.\n"
        )
        return prompt