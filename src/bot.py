'''A bot that does one shot analysis.'''
import os
from xmlrpc import client
import src.constants as cn

from google import genai # type: ignore
import os
import pandas as pd  # type: ignore
from scipy.stats import chi2_contingency  # type: ignore
from sklearn.metrics import roc_auc_score # type: ignore
from typing import List, Tuple, cast

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

    def __init__(self, path:str=cn.MERGED_DATA_PTH,
            selected_columns:List[str]=["pathology_report"],
            model="gemini-2.5-flash",
            key_path="/Users/jlheller/google_api_key_paid.txt") -> None:
        """
        Args:
            path (str, optional): _description_. Defaults to cn.MERGED_DATA_PTH.
                CSV file
            selected_columns (List[str], optional): _description_. Defaults to
                ["cases.submitter_id", "pathology_report"].
            model (str, optional): _description_. Defaults to "gemini-2.5-flash".

        """
        self.key_path = key_path
        self.path = path
        self.model = model
        self.full_data_df= pd.read_csv(path)
        self.columns = self.full_data_df.columns.tolist()
        if not set(selected_columns).issubset(set(self.columns)):
            raise ValueError(f"Selected columns not in {path}")
        self.selected_columns = selected_columns
        self.selected_data_df = self.full_data_df[selected_columns]
        self._initializeEnvironment()
        self.oneshot_results: list = []
        self.one_shot_idx = 0 # index to keep track of one shot analyses
        self.client = genai.Client()


    def _initializeEnvironment(self) -> None:
        with open(self.key_path, "r") as f:
            gemini_api_key = f.read()
        os.environ["GEMINI_API_KEY"] = gemini_api_key

    def makeChat(self):
        '''Make a chat object.'''
        chat = self.client.chats.create(model=self.model)
        return chat
    
    def executeOneshot(self)->None:
        '''Do one shot analysis.
        '''
        chat = self.makeChat()
        prompt_data = ""
        for column in self.selected_columns:
            prompt_data += f"{column}: {self.selected_data_df.loc[self.one_shot_idx][column]}\n"
        prompt = ONESHOT_PROMPT % prompt_data
        response = chat.send_message(prompt)
        if response is None:
            raise ValueError("No response from Gemini")
        extracted_response = float(response.text) # type: ignore
        self.oneshot_results.append(extracted_response)
        self.one_shot_idx += 1

    def calculateAUC(self)->float:
        '''Calculate AUC for one shot results.
        '''
        true_binary_labels = self.full_data_df['OS'].tolist()[0:len(self.oneshot_results)] 
        auc = roc_auc_score(true_binary_labels, self.oneshot_results)
        return cast(float, auc)