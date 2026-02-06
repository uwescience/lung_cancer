# Prompts used in prediction
* zeroshot_single
    * Prediction for a single patient. Prompt embeds a ``%s`` that is substituted for column data
* zeroshot_batach
    * Prompts for a batch of zero shot predictions
* Prompt files are python modules that define the function ``getPrompt`` which returns a string. For single zeroshot, this string is a text
pattern that expects patient information.
* Prompt files must be in the ``prompt`` directory within the appropriate subdirectory (zeroshot_single, zeroshot_batch).
