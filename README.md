# NLP at UC Santa Cruz at SemEval-2024 Task 5: Legal Answer Validation using Few-Shot Multi-Choice QA

This repository contains the code for our submission for [SemEval-2024 Task 5: The Legal Argument Reasoning Task in Civil Procedure](https://trusthlt.github.io/semeval24/). Our submission secured the 7th place on the leaderboard. The paper published in conjunction with this codebase can be found here: [NLP at UC Santa Cruz at SemEval-2024 Task 5: Legal Answer Validation using Few-Shot Multi-Choice QA](https://aclanthology.org/2024.semeval-1.189/).

## Contained Files

**Note: When we refer to data as being in "binary classification format", we mean that it is in the format provided to us by the task organizers. When we refer to it being in "multi-choice format", we mean that we have run the reformatting script on the data provided to us to create a new dataset.**

### GPT

- `binary_few_shot.py`: Performs few shot prompting for either GPT-3.5 or GPT-4 on the test data. Data must be in binary classification format.
- `multi_choice_few_shot.py`: Performs few shot prompting for either GPT-3.5 or GPT-4 on the test data. Data must be in multi-choice format.
- `reformat_test_data.py`: Converts test data from binary classification format to multi-choice format.

### BERT

- `inference.py`: Generates predictions for the intended BERT model.
- `train.py`: Runs training on the intended BERT model. 


## How to Run

### GPT

- Make sure the environment you are running the scripts in has an OpenAI access key defined so that the API can be accessed. 
- To run few shot prompting with binary classification, use `binary_few_shot.py`.
- To run few shot prompting with multi-choice classification, first reformat the data using `reformat_test_data.py`. Then, use `multi_choice_few_shot.py`.
- For both scripts, set the `model` variable to the desired version of GPT-3.5 or GPT-4, then you may run the script.

### BERT

- There is a collection of bash scripts that when run, call either `train.py` or `inference.py` for either vanilla BERT or Legal BERT.
- If need be, change the `--dataset` flag in the bash script to the appropriately named dataset you have stored locally.
