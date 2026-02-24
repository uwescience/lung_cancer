'''A bot that generates and analyzes survival predictions.'''

"""
Notes
    1. Prompt files are in prompts/zeroshot_single and prompts/zeroshot_batch directories. Each file contains a getPrompt function that returns the
        prompt pattern for the single zero-shot and batch zero-shot analyses, respectively.
"""


from http import client
import os
import time
import src.constants as cn
from src.multishot_maker import MultishotMaker

from google import genai # type: ignore
from google.genai import types # type: ignore
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
BATCH_DIR = "batch"


class Bot(object):
    '''A bot that collects survival data'''

    def __init__(self, diagnostic_pth:str=cn.MERGED_DATA_PTH,
            selected_columns:List[str]=["cases.submitter_id", "pathology_report"],
            model="gemini-2.5-flash",
            key_path="/Users/jlheller/google_api_key_paid.txt",
            experiment_filename: Optional[str]=None,
            experiment_dir:str = cn.EXPERIMENT_DIR,
            is_initialize_experiment_file: bool=False,
            is_randomized: bool=False,
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
            is_randomized (bool, optional): If True, randomizes the order of predictor columns
            is_mock (bool, optional): If True, uses mock responses for testing.

        """
        self.is_mock = is_mock
        self.is_randomized = is_randomized
        if experiment_filename is None:
            experiment_filename = str(np.random.randint(1000000, 9999999)) + ".csv"
        self.experiment_filename = experiment_filename
        self.experiment_pth = os.path.join(experiment_dir, self.experiment_filename)
        if os.path.exists(self.experiment_pth) and is_initialize_experiment_file:
            os.remove(self.experiment_pth)
        df = pd.read_csv(self.experiment_pth) if os.path.exists(self.experiment_pth) else pd.DataFrame()
        self.zeroshot_idx = len(df) # index to keep track of zero shot analyses
        self.key_path = key_path
        self.path = diagnostic_pth
        self.model = model
        self.full_data_df= pd.read_csv(diagnostic_pth)
        self.full_data_df[cn.COL_UNIQUE_ID] = range(len(self.full_data_df))
        self.data_len = len(self.full_data_df.index)
        self.columns = self.full_data_df.columns.tolist()
        if not set(selected_columns).issubset(set(self.columns)):
            raise ValueError(f"Selected columns not in {diagnostic_pth}")
        self.selected_columns = list(selected_columns)
        self.selected_columns.remove(cn.COL_SUBMITTER_ID) if cn.COL_SUBMITTER_ID in self.selected_columns else None
        self.selected_columns.append(cn.COL_UNIQUE_ID)
        self.selected_data_df = self.full_data_df[self.selected_columns]
        if self.is_randomized:
            for column in self.selected_columns:
                self.selected_data_df[column] = np.random.permutation(self.selected_data_df[column])
        self._initializeEnvironment()
        self.client = genai.Client()
        self.generation_config = types.GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
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
        chat = self.client.chats.create(model=self.model, config=self.generation_config)
        return chat
    
    def executeSingleZeroshot(self, data_idx:int=0, prompt_file:str="prompt1.py")->dict:
        '''Builds and submits the prompt for zero shot analysis. Uses a new chat.
        Args:
            data_idx (int): Index of the data row to analyze.
                Defaults to 0.
            prompt_file (str, optional): Name of file in the prompt/zeroshot_batch directory
                containing the prompt to use for the single zero-shot analysis.
                Defaults to "prompt1.py".
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
        prompt = self._getPrompt(prompt_file=prompt_file, directory="zeroshot_single") % prompt_data
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

    def executeMultipleSingleZeroshot(self, num_shot:int, prompt_file:str="prompt1.py")->pd.DataFrame:
        """Executes multiple zero-shot analyses in sequence,
            saving results to the experiment file.
        Args:
            num_shot (int): Number of zero-shot analyses to execute.

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
            if self.zeroshot_idx >= self.data_len:
                break
            result_dct = self.executeSingleZeroshot(self.zeroshot_idx,
                    prompt_file=prompt_file)
            result_dcts.append(result_dct)
            self.zeroshot_idx += 1
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
        """Plot ROC curve for zero shot results.

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

    @classmethod
    def plotROCs(cls,
            result_dir_names: List[str],
            experiment_dir_pth: Optional[str]=None,
            figsize=(8,6),
            legends:Optional[List[str]]=None,
            is_plot:bool = True)-> None:
        """Plot ROC curves for multiple experiment files on the same plot.

        Args:
            result_dir_names (List[str]): List of directory names for experiment files containing CSV files.
            experiment_dir_pth (Optional[str], optional): Path to experiment directory.
            figsize (tuple, optional): Figure size. Defaults to (8,6).
            legends (Optional[List[str]], optional): List of legends for each experiment file. Defaults to None.
            is_plot (bool, optional): If True, shows the plot. Defaults to True.
        """
        all_result_dct: Dict[str, pd.DataFrame] = {}
        for result_dir_name in result_dir_names:
            dct = cls.getExperimentResults(result_dir_name,
                experiment_dir_pth=experiment_dir_pth)
            all_result_dct.update(dct)
        keys = list(all_result_dct.keys())
        # Construct the median value
        all_df = pd.concat([all_result_dct[f] for f in all_result_dct], ignore_index=True)
        dfg = all_df.groupby(cn.COL_UNIQUE_ID)
        medians = dfg[cn.COL_PREDICTED].median().tolist()
        median_df = pd.DataFrame(medians, columns=[cn.COL_PREDICTED])
        median_df[cn.COL_UNIQUE_ID] = dfg[cn.COL_UNIQUE_ID].first().tolist()
        median_df.set_index(cn.COL_UNIQUE_ID, inplace=True)
        median_df[cn.COL_ACTUAL] = dfg[cn.COL_ACTUAL].first().tolist()
        all_result_dct["Median Prediction"] = median_df
        # Plot ROC curve for each file
        plt.figure(figsize=figsize)
        for idx, (filename, df) in enumerate(all_result_dct.items()):
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
            label = legends[idx] if legends is not None and idx < len(legends) else filename
            plt.plot(fpr, tpr, lw=2,
                label=f'{label} (AUC = {roc_auc:.2f})')

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
        maxs = df.groupby(cn.COL_UNIQUE_ID)[cn.COL_PREDICTED].max().tolist()
        mins = df.groupby(cn.COL_UNIQUE_ID)[cn.COL_PREDICTED].min().tolist()
        ranges = [maxs[i] - mins[i] for i in range(len(maxs))]
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
            prompt,
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
                    contents=[prompt, uploaded_file],
                    config=self.generation_config)
            response_text = response.text
            # Clean the response text
        else:
            unique_ids = dataframe[cn.COL_UNIQUE_ID].tolist()
            with open(LOCAL_CONTEXT_FILE, "r") as f:
                file_content = f.readlines()
            length = len(file_content) - 1 # exclude header
            response_text = "\n".join(
                    [f"{unique_ids[n]},{str(np.random.uniform(0, 1))}"
                    for n in range(length)])
        # Add the header if missing
        response_text = str(response_text).strip()
        if not cn.COL_UNIQUE_ID in response_text:
            response_text = f"{cn.COL_UNIQUE_ID},{cn.COL_PREDICTED}\n" + response_text
        #
        return response_text, response  # type: ignore
    
    def _getPrompt(self, prompt_file:str="prompt1.py", directory:str="batch")->str:
        """Gets the batch prompt from the specified file.

        Args:
            prompt_file (str, optional): Name of file in the prompt/zeroshot_batch directory containing the prompt to use for the batch zero-shot analysis. Defaults to "prompt1.py".
            directory (str, optional): Name of directory in the prompts folder containing the prompt file. Defaults to "zeroshot_batch".

        Returns:
            str: The batch prompt.
        """
        prompt_module = __import__(f"prompts.{directory}.{prompt_file[:-3]}", fromlist=['getPrompt'])
        prompt = prompt_module.getPrompt()
        return prompt

    def executeBatchMultishot(self, prompt_file:str="prompt1.py",
            num_example:int=0)->pd.DataFrame:
        '''Uploads the data for multiple multi-shot analyses in batches. Then submits the prompt.

        Args:
            prompt_file (str):  Name of file in the prompt/batch directory containing the prompt to use for the batch zero-shot analysis.
            num_example (int): Number of examples to include in the prompt. If 0, includes all examples.

        Returns:
            pd.DataFrame:
                <column>: column in prompt (str)
                predicted: returned from LLM (float)
                actual: true label (float)
        '''
        # Initializaitons
        MAX_RETRIES = 10 
        all_response_df = pd.DataFrame()
        unprocessed_patients = self.selected_data_df[cn.COL_UNIQUE_ID].tolist()
        prev_patient_count = len(unprocessed_patients)
        # Construct the examples
        if num_example == 0:
            example_str = ""
        else:
            multishot_maker = MultishotMaker(self.full_data_df, num_example=num_example)
            example_str, example_case_ids = multishot_maker.buildExamples()
            unprocessed_patients = [p for p in unprocessed_patients if p not in example_case_ids]
        # Process until all patients are done
        result_df = pd.DataFrame()
        for _ in range(MAX_RETRIES):
            if len(unprocessed_patients) == 0:
                break
            df = self.selected_data_df[
                self.selected_data_df[cn.COL_UNIQUE_ID].isin(unprocessed_patients)]
            prompt = self._getPrompt(prompt_file=prompt_file, directory=BATCH_DIR)
            # TO DO: Add the examples to the prompt in a more principled way
            prompt = prompt + "\n" + example_str
            response_text, _ = self._executeGenerateContent(prompt=prompt, dataframe=df)
            # Create the response dataframe
            try:
                response_df = pd.read_csv(StringIO(response_text))
            except Exception as e:
                print(f"Quitting because error reading response text: {e}")
                break
            response_df = response_df[
                    response_df[cn.COL_UNIQUE_ID].isin(unprocessed_patients)]
            response_df = response_df.rename(columns={response_df.columns[1]: cn.COL_PREDICTED}) # type: ignore
            columns = response_df.columns.tolist()
            columns[1] = cn.COL_PREDICTED
            response_df.columns = columns
            # Eliminate redunant responses
            response_df = response_df.groupby(cn.COL_UNIQUE_ID).mean().reset_index()
            # Eliminate processed patients
            unprocessed_patients = [p for p in unprocessed_patients
                if p not in response_df[cn.COL_UNIQUE_ID].tolist()]
            all_response_df = pd.concat([all_response_df, response_df], ignore_index=True)
            # Check for progress
            if len(unprocessed_patients) == prev_patient_count:
                raise RuntimeError("No progress made in processing patients")
            prev_patient_count = len(unprocessed_patients)
            result_df = pd.merge(all_response_df, 
                                self.full_data_df[[cn.COL_UNIQUE_ID, 'OS']],
                                on=cn.COL_UNIQUE_ID,
                                how='left')
            result_df.rename(columns={'OS': cn.COL_ACTUAL}, inplace=True)
            result_df.to_csv(self.experiment_pth, index=False)
        if len(unprocessed_patients) > 0:
            print(f"Warning: Unprocessed patients remaining after {MAX_RETRIES} retries: {unprocessed_patients}")
        # Join with the original data to get actual labels
        return result_df