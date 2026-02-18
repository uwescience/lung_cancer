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

    def __init__(self, data_df: pd.DataFrame, num_examples: int = 4) -> None:
        """
        Args:
            data_df (pd.DataFrame): DataFrame of patients.
            num_examples (int): Number of example patients to select.
                Must be a multiple of 4 (one survivor and one non-survivor
                per cancer type per round).
        """
        if num_examples < 4 or num_examples % 4 != 0:
            raise ValueError(
                f"num_examples must be a positive multiple of 4, got {num_examples}")
        self.data_df = data_df
        self.num_examples = num_examples
        self.example_df = self._chooseExamples()

    def _chooseExamples(self) -> pd.DataFrame:
        """Chooses example patients: equal numbers of survivors and
        non-survivors for each of adenocarcinoma and squamous cell cancer.

        Returns:
            pd.DataFrame: num_examples rows with columns from the original dataset.
        """
        per_group = self.num_examples // 4
        selections = []
        for disease_type in [DISEASE_ADENOCARCINOMA, DISEASE_SQUAMOUS]:
            disease_df = self.data_df[self.data_df[COL_DISEASE_TYPE] == disease_type]
            for outcome in [1, 0]:
                candidate_df = disease_df[disease_df[COL_OS] == outcome]
                if len(candidate_df) < per_group:
                    raise ValueError(
                        f"Need {per_group} patients for {disease_type} with "
                        f"OS={outcome}, but only {len(candidate_df)} available")
                row_df = candidate_df.sample(n=per_group, random_state=None)
                selections.append(row_df)
        result_df = pd.concat(selections)
        self._example_indices = result_df.index
        return result_df.reset_index(drop=True)

    def buildPrompt(self) -> tuple[str, pd.DataFrame]:
        """Builds a few-shot prompt using the example patients.
        The prompt uses only the pathology report as the example data.

        Returns:
            tuple: (prompt string with %%s placeholder, DataFrame of remaining patients)
        """
        outcome_labels = {1: "survive", 0: "not_survive"}
        examples_text = ""
        for idx, row in self.example_df.iterrows():
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
        remaining_df = self.data_df.drop(self._example_indices)
        return prompt, remaining_df