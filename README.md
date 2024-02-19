# NLP at UC Santa Cruz at SemEval-2024 Task 5: Legal Answer Validation using Few-Shot Multi-Choice QA

This repository contains the codebase that underlies our experimentation for the Task. 

## Contained Files

**Data: When we refer to data as being in "binary classification format", we mean that it is in the format provided to us by the task organizers. When we refer to it being in "multi-choice format", we mean that we have run the reformatting script on the data provided to us to create a new dataset.**

### GPT

- `binary_few_shot.py`: Performs few shot prompting for either GPT-3.5 or GPT-4 on the test data. Data must be in binary classification format.
- `multi_choice_few_shot.py`: Performs few shot prompting for either GPT-3.5 or GPT-4 on the test data. Data must be in multi-choice format.
- `reformat_tesT_data.py`: Converts test data from binary classification format to multi-choice format.

### BERT

