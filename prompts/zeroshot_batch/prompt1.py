import src.constants as cn

import os

LOCAL_CONTEXT_FILE = os.path.join(cn.DATA_DIR, "local_context.csv")

def getPrompt(file_path:str=LOCAL_CONTEXT_FILE)->str:
    prompt = f"""
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
    return prompt