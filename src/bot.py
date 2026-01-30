'''A bot that does one shot analysis.'''
from http import client
import os
import time
import src.constants as cn

from google import genai # type: ignore
import os
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import roc_auc_score # type: ignore
from typing import List, cast, Optional, Dict, Any
from sklearn.metrics import roc_curve, auc  # type: ignore
from sklearn.metrics import RocCurveDisplay  # type: ignore
from io import StringIO

LOCAL_CONTEXT_FILE = os.path.join(cn.DATA_DIR, "local_context.csv")
ONESHOT_PROMPT = """
Instruction: You are a clinical oncologist with expertise in cancer prognosis.

Task: Based on the following pathology report, predict whether the patient survived
beyond 2 years from the date of diagnosis.

%s

Output format (no explanation):
indicate the probability of a 2 year survival. Only return a probability value between 0 and 1
"""
ONESHOT_FILE_PROMPT = f"""
Instruction: You are a clinical oncologist with expertise in cancer prognosis.

Task: Using the file {LOCAL_CONTEXT_FILE}, predict whether the patient survived
beyond 2 years from the date of diagnosis. Each row in the file is a different patient.
So, you are processing a batch of requests. Provide a response for each row in the file.
Do not skip any rows.
The columns are as follows:
  *cases.submitter_id: Unique patient identifier
  *pathology_report: Text of the pathology report

Output format (no explanation):
indicate the probability of a 2 year survival.
Only return a probability value between 0 and 1
"""


class Bot(object):
    '''A bot that does one collects survival data'''

    def __init__(self, diagnostic_pth:str=cn.MERGED_DATA_PTH,
            selected_columns:List[str]=["cases.submitter_id", "pathology_report"],
            model="gemini-2.5-flash",
            key_path="/Users/jlheller/google_api_key_paid.txt",
            experiment_filename: Optional[str]=None,
            experiment_dir:str = cn.EXPERIMENT_DIR,
            is_initialize_experiment_file: bool=False,
            is_mock: bool=False) -> None:
        """
        Collects survival data. If the experiment_filename is provided,
        saves results to that file and resumes from previous results.

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
            experiment_dir (str, optional): Directory for experiment results.
            is_initialize (bool, optional): If True, initializes the experiment file.
            is_mock (bool, optional): If True, uses mock responses for testing.

        """
        self.is_mock = is_mock
        if experiment_filename is None:
            experiment_filename = str(np.random.randint(1000000, 9999999)) + ".csv"
        self.experiment_filename = experiment_filename
        self.experiment_pth = os.path.join(experiment_dir, self.experiment_filename)
        if os.path.exists(self.experiment_pth) and is_initialize_experiment_file:
            os.remove(self.experiment_pth)
        df = pd.read_csv(self.experiment_pth) if os.path.exists(self.experiment_pth) else pd.DataFrame()
        self.oneshot_idx = len(df) # index to keep track of one shot analyses
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
        self.client = genai.Client()
        self.uploaded_file_dct: dict = {}

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
            try:
                result_dct[cn.COL_PREDICTED] = float(response.text) # type: ignore
            except Exception as e:
                import pdb; pdb.set_trace()
        #
        result_dct[cn.COL_ACTUAL] = self.full_data_df.loc[data_idx, 'OS']
        return result_dct

    def executeMultipleOneshot(self, num_shot:int)->pd.DataFrame:
        """Executes multiple one-shot analyses in sequence,
            saving results to the experiment file.
        Args:
            num_shot (int): Number of one-shot analyses to execute.

        Returns:
            pd.DataFrame:
                <column>: column in prompt (str)
                predicted: returned from LLM (float)
                actual: true label (float)
        """
        # Initializaitons
        # Execute
        result_dcts: list = []
        for _ in range(num_shot):
            if self.oneshot_idx >= self.data_len:
                break
            result_dct = self.executeOneshot(self.oneshot_idx)
            result_dcts.append(result_dct)
            self.oneshot_idx += 1
        # Convert to a dict of lists
        # Save results
        if (len(result_dcts) > 0):
            result_dct = {key: [d[key] for d in result_dcts] for key in result_dcts[0]}
            result_df = pd.DataFrame(result_dct)
            if os.path.exists(self.experiment_pth):
                previous_results_df = pd.read_csv(self.experiment_pth)
            else:
                previous_results_df = pd.DataFrame()
            full_result_df = pd.concat([previous_results_df, result_df], ignore_index=True)
            full_result_df.to_csv(self.experiment_pth, index=False)
        else:
            result_df = pd.DataFrame()
        return result_df
    
    @staticmethod
    def plotROC(experiment_df:pd.DataFrame)->None:
        """Plot ROC curve for one shot results.

        Args:
            result_df (pd.DataFrame): DataFrame with results.  
        """
        from sklearn.metrics import RocCurveDisplay # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        true_binary_labels = experiment_df[cn.COL_ACTUAL].tolist()[0:len(experiment_df)] 
        RocCurveDisplay.from_predictions(
            true_binary_labels,
            experiment_df[cn.COL_PREDICTED].tolist()
        )
        plt.show()

    #def plotROCs(cls, directory_path: str, figsize=(8,6)) -> None:
    @classmethod
    def plotROCs(cls, result_dir_name: str,
            experiment_dir_pth: Optional[str]=None, figsize=(8,6),
            is_plot:bool = True)-> None:
        """Plot ROC curves for multiple experiment files on the same plot.

        Args:
            directory_path (str): Path to directory containing experiment CSV files.
        """
        result_dct = cls.getExperimentResults(result_dir_name,
            experiment_dir_pth=experiment_dir_pth)
        keys = list(result_dct.keys())
        # Construct the median value
        all_df = pd.concat([result_dct[f] for f in result_dct], ignore_index=True)
        dfg = all_df.groupby(cn.COL_SUBMITTER_ID)
        medians = dfg[cn.COL_PREDICTED].median().tolist()
        median_df = pd.DataFrame(medians, columns=[cn.COL_PREDICTED])
        median_df[cn.COL_SUBMITTER_ID] = dfg[cn.COL_SUBMITTER_ID].first().tolist()
        median_df.set_index(cn.COL_SUBMITTER_ID, inplace=True)
        median_df[cn.COL_ACTUAL] = dfg[cn.COL_ACTUAL].first().tolist()
        result_dct["Median Prediction"] = median_df
        # Plot ROC curve for each file
        for filename, df in result_dct.items():
            # Extract predicted and actual columns
            if cn.COL_PREDICTED not in df.columns or cn.COL_ACTUAL not in df.columns:
                print(f"Warning: Skipping {filename} - missing 'predicted' or 'actual' columns")
                continue
            y_true = df[cn.COL_ACTUAL].tolist()
            y_scores = df[cn.COL_PREDICTED].tolist()
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(fpr, tpr, lw=2, label=f'{filename} (AUC = {roc_auc:.2f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        # Labels and formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multiple Experiments')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        if is_plot:
            plt.show()

    @classmethod
    def getExperimentResults(cls, result_dir_name: str,
            experiment_dir_pth: Optional[str]=None)-> Dict[str, pd.DataFrame]:
        """Gets the experiment results from a directory of results.

        Args:
            result_dir_name (Optional[str], optional): Directory name for experiment file
                in experiment directory. Defaults to None.
            experiment_dir_pth (Optional[str], optional): Path to experiment directory.

        Returns:
            Dict[str, pd.DataFrame]: 
                key: filename
                value: DataFrame with results

        """
        result_dct: Dict[str, pd.DataFrame] = {}
        if experiment_dir_pth is None:
            experiment_dir_pth = cn.EXPERIMENT_DIR
        # Build the dataframe dictionary
        dir_path = os.path.join(experiment_dir_pth, result_dir_name) if result_dir_name is not None else None
        if dir_path is None:
            raise ValueError("dir_name must be provided if result_df is empty")
        experiment_files = os.listdir(dir_path)
        experiment_paths = [os.path.join(dir_path, f) for f in experiment_files
                if f.endswith(".csv")]
        for idx, path in enumerate(experiment_paths):
            if not os.path.isfile(path):
                raise RuntimeError(f"File not found: {path}")
            result_dct[experiment_files[idx]] = pd.read_csv(path)
        return result_dct
    
    @classmethod
    def plotPredictionRange(cls, result_dir_name: str,
            experiment_dir_pth: Optional[str]=None,
            is_plot: bool = True)-> None:
        """Plots a histogram of the ranges of predictions within the directory.

        Args:
            result_dir_name (str): directory with replications
            experiment_dir_pth (Optional[str], optional): Path to experiment directory. Defaults to None.
        """
        result_dct = cls.getExperimentResults(result_dir_name,
            experiment_dir_pth=experiment_dir_pth)
        df = pd.concat([result_dct[f] for f in result_dct], ignore_index=True)
        stds = df.groupby(cn.COL_SUBMITTER_ID)[cn.COL_PREDICTED].std().tolist()
        maxs = df.groupby(cn.COL_SUBMITTER_ID)[cn.COL_PREDICTED].max().tolist()
        mins = df.groupby(cn.COL_SUBMITTER_ID)[cn.COL_PREDICTED].min().tolist()
        ranges = [maxs[i] - mins[i] for i in range(len(maxs))]
        #plt.hist(stds, bins=30, alpha=0.7, density=True)
        #plt.hist(ranges, bins=30, alpha=0.7, culmulative=True, density=True)
        ranges.sort()
        x_arr = np.array(ranges)
        y_arr = np.array(range(len(ranges))) / len(ranges)
        plt.plot(x_arr, y_arr)
        plt.xlabel("Range of Predictions")
        plt.ylabel("Fraction of Samples")
        plt.title("Distribution of Range of Survival Predictions for the Same Patient")
        if is_plot:
            plt.show()

    def _executeGenerateContent(self, 
            prompt=ONESHOT_FILE_PROMPT,
            dataframe:pd.DataFrame=pd.DataFrame()
            )-> tuple[str, Any]:
        """Uploads the file and obtains the response.

        Args:
            prompt (_type_, optional): _description_. Defaults to ONESHOT_FILE_PROMPT.
            dataframe (Optional[pd.DataFrame], optional): DataFrame to use for the prompt

        Returns:
            tuple[str, Optional[genai.client.models.Response]]: _description_
        """
        # Upload the file
        dataframe.to_csv(LOCAL_CONTEXT_FILE, index=False)
        uploaded_file = self.client.files.upload(file=LOCAL_CONTEXT_FILE)
        # Wait for file processing
        while uploaded_file.state == "PROCESSING":
            time.sleep(1)
            uploaded_file = self.client.files.get(name=uploaded_file.name) # type: ignore
        # Get the response
        response = None
        if not self.is_mock:
            response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, uploaded_file])
            response_text = response.text
            # Clean the response text
        else:
            submitter_ids = self.full_data_df[cn.COL_SUBMITTER_ID].tolist()
            with open(LOCAL_CONTEXT_FILE, "r") as f:
                file_content = f.readlines()
            length = len(file_content) - 1 # exclude header
            response_text = "\n".join(
                    [f"{submitter_ids[n]},{str(np.random.uniform(0, 1))}"
                    for n in range(length)])
        # Add the header if missing
        response_text = str(response_text).strip()
        if not cn.COL_SUBMITTER_ID in response_text:
            response_text = f"{cn.COL_SUBMITTER_ID},{cn.COL_PREDICTED}\n" + response_text
        #
        return response_text, response  # type: ignore

    def executeMultipleOneshotInFile(self, batch_size:int=50)->pd.DataFrame:
        '''Uploads the data for multiple one-shot analyses in batches. Then submits the prompt.

        Args:
            batch_size (int): Number of rows to process in each batch. Defaults to 50.

        Returns:
            pd.DataFrame:
                <column>: column in prompt (str)
                predicted: returned from LLM (float)
                actual: true label (float)
        '''
        all_response_df = pd.DataFrame()
        unprocessed_patients = self.selected_data_df[cn.COL_SUBMITTER_ID].tolist()
        prev_patient_count = len(unprocessed_patients)
        # Process until all patients are done
        result_df = pd.DataFrame()
        while len(unprocessed_patients) > 0:
            df = self.selected_data_df[
                self.selected_data_df[cn.COL_SUBMITTER_ID].isin(unprocessed_patients)]
            response_text, response = self._executeGenerateContent(dataframe=df)
            if len(df) == 0:
                import pdb; pdb.set_trace()
            # Create the response dataframe
            try:
                response_df = pd.read_csv(StringIO(response_text))
            except Exception as e:
                print(f"Quitting because error reading response text: {e}")
                break
            response_df = response_df[
                    response_df[cn.COL_SUBMITTER_ID].isin(unprocessed_patients)]
            columns = response_df.columns.tolist()
            columns[1] = cn.COL_PREDICTED
            response_df.columns = columns
            # Eliminate redunant responses
            try:
                response_df = response_df.groupby(cn.COL_SUBMITTER_ID).mean().reset_index()
            except Exception as e:
                import pdb; pdb.set_trace()
            # Eliminate processed patients
            unprocessed_patients = [p for p in unprocessed_patients
                if p not in response_df[cn.COL_SUBMITTER_ID].tolist()]
            all_response_df = pd.concat([all_response_df, response_df], ignore_index=True)
            # Check for progress
            if len(unprocessed_patients) == prev_patient_count:
                raise RuntimeError("No progress made in processing patients")
            prev_patient_count = len(unprocessed_patients)
            result_df = pd.merge(all_response_df, 
                                self.full_data_df[[cn.COL_SUBMITTER_ID, 'OS']], 
                                on=cn.COL_SUBMITTER_ID,
                                how='left')
            result_df.rename(columns={'OS': cn.COL_ACTUAL}, inplace=True)
            result_df.to_csv(self.experiment_pth, index=False)
        # Join with the original data to get actual labels
        return result_df