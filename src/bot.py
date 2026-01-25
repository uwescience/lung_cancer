'''A bot that does one shot analysis.'''
import os
import src.constants as cn

from google import genai # type: ignore
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import roc_auc_score # type: ignore
from typing import List, cast, Optional

ONESHOT_PROMPT = """
Instruction: You are a clinical oncologist with expertise in cancer prognosis.

Task: Based on the following pathology report, predict whether the patient survived
beyond 2 years from the date of diagnosis.

%s

Output format (no explanation):
indicate the probability of a 2 year survival. Only return a probability value between 0 and 1
"""


class Bot(object):
    '''A bot that does one collects survival data'''

    def __init__(self, diagnostic_pth:str=cn.MERGED_DATA_PTH,
            selected_columns:List[str]=["cases.submitter_id", "pathology_report"],
            model="gemini-2.5-flash",
            key_path="/Users/jlheller/google_api_key_paid.txt",
            experiment_filename: Optional[str]=None,
            is_mock: bool=False) -> None:
        """
        Args:
            diagnostic_pth (str, optional): _description_. Defaults to cn.MERGED_DATA_PTH.
                CSV file
            selected_columns (List[str], optional): _description_. Defaults to
                ["cases.submitter_id", "pathology_report"].
            model (str, optional): _description_. Defaults to "gemini-2.5-flash".
            key_path (str, optional): _description_. Defaults to
                "/Users/jlheller/google_api_key_paid.txt".
            experiment_filename (Optional[str], optional): Name of CSV file for experiment results
                Defaults to None.
            is_mock (bool, optional): If True, uses mock responses for testing.

        """
        self.is_mock = is_mock
        if experiment_filename is None:
            experiment_filename = str(np.random.randint(1000000, 9999999)) + ".csv"
        self.experiment_filename = experiment_filename
        self.experiment_path = os.path.join(cn.EXPERIMENT_DIR, self.experiment_filename)
        self.key_path = key_path
        self.path = diagnostic_pth
        self.model = model
        self.full_data_df= pd.read_csv(diagnostic_pth)
        self.data_len = len(self.full_data_df.index)
        self.columns = self.full_data_df.columns.tolist()
        if not set(selected_columns).issubset(set(self.columns)):
            raise ValueError(f"Selected columns not in {diagnostic_pth}")
        self.selected_columns = selected_columns
        self.selected_data_df = self.full_data_df[selected_columns]
        self._initializeEnvironment()
        self.oneshot_idx = 0 # index to keep track of one shot analyses
        self.client = genai.Client()

    def getExperimentFilename(self)->str:
        '''Get the experiment filename.
        Returns:
            str: Experiment filename.
        '''
        return self.experiment_filename

    def _initializeEnvironment(self) -> None:
        with open(self.key_path, "r") as f:
            gemini_api_key = f.read()
        os.environ["GEMINI_API_KEY"] = gemini_api_key

    def makeChat(self):
        '''Make a chat object.'''
        chat = self.client.chats.create(model=self.model)
        return chat
    
    def executeOneshot(self, data_idx:int=0)->dict:
        '''Builds and submits the prompt for one shot analysis. Uses a new chat.
        Args:
            data_idx (int): Index of the data row to analyze.
                Defaults to 0.
        Returns:
            dict:
                <column>: column in prompt (str)
                predicted: returned from LLM (float)
                actual: true label (float)
        '''
        if data_idx >= self.data_len:
            raise IndexError("data_idx out of range")
        # Initialize
        chat = self.makeChat()
        result_dct:dict = {}
        # Construct the prompt
        prompt_data = ""
        for column in self.selected_columns:
            result_dct[column] = self.selected_data_df.loc[data_idx][column]
            prompt_data += f"{column}: {result_dct[column]}\n"
        prompt = ONESHOT_PROMPT % prompt_data
        # Get the response
        if self.is_mock:
            # For testing, return a random prediction
            result_dct[cn.COL_PREDICTED] = float(np.random.uniform(0, 1))
        else:
            response = chat.send_message(prompt)
            if response is None:
                raise ValueError("No response from Gemini")
            result_dct[cn.COL_PREDICTED] = float(response.text) # type: ignore
        #
        result_dct[cn.COL_ACTUAL] = self.full_data_df.loc[data_idx, 'OS']
        return result_dct

    def executeMultipleOneshot(self, num_shot:int, is_initialize:bool = False)->pd.DataFrame:
        """Executes multiple one-shot analyses in sequence,
            saving results to the experiment file.
        Args:
            num_shot (int): _description_
            is_initialize (bool, optional): If True, resets the index to 0.
                Defaults to False.

        Returns:
            pd.DataFrame:
                <column>: column in prompt (str)
                predicted: returned from LLM (float)
                actual: true label (float)
        """
        # Initializaitons
        if is_initialize:
            self.oneshot_idx = 0
            os.remove(self.experiment_path) if os.path.exists(self.experiment_path) else None
        # Execute
        result_dcts: list = []
        for _ in range(num_shot):
            if self.oneshot_idx >= self.data_len:
                break
            result_dct = self.executeOneshot(self.oneshot_idx)
            result_dcts.append(result_dct)
            self.oneshot_idx += 1
        # Convert to a dict of lists
        result_dct = {key: [d[key] for d in result_dcts] for key in result_dcts[0]}
        # Save results
        result_df = pd.DataFrame(result_dct)
        if (len(result_df) > 0):
            if os.path.exists(self.experiment_path):
                previous_results_df = pd.read_csv(self.experiment_path)
            else:
                previous_results_df = pd.DataFrame()
            full_result_df = pd.concat([previous_results_df, result_df], ignore_index=True)
            full_result_df.to_csv(self.experiment_path, index=False)
        return result_df

    def calculateAUC(self)->float:
        '''Calculate AUC for one shot results.
        '''
        full_result_df = pd.read_csv(self.experiment_path)
        true_binary_labels = full_result_df[cn.COL_ACTUAL].tolist()[0:len(full_result_df)] 
        auc = roc_auc_score(true_binary_labels, full_result_df[cn.COL_PREDICTED].tolist())
        return cast(float, auc)