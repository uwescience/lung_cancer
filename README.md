# lung_cancer
Analysis of lung cancer data for Winter 2026 accelerator

## Configuration
I could run the code without any issues. I modified the script to remove the submitter ID in the prompt. And I used the following model config to enforce deterministic output: config = {"temperature": 0.0,"top_p": 1.0,"top_k": 1,}. However, the script gave different results when running two tests (batch_size = 1, num_batch = 7) vs. (batch_size = 7, num_batch = 1). I thought they should give the same results as the chatbot was initiated for each report. The reason may be that there is still some randomness even with that config file.
