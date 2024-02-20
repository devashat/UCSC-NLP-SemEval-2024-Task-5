import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import argparse
import zipfile


def init(pretrained_model, load_checkpoint_dir):


    if load_checkpoint_dir:
        model = AutoModelForSequenceClassification.from_pretrained(load_checkpoint_dir)
        print(f'Loading model from checkpoint: {load_checkpoint_dir}')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        print(f'Loading model from huggingface: {pretrained_model}')        
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    return model, tokenizer

def preprocess_test_data(question, answer, tokenizer):

    input_text = f"{question} [SEP] {answer}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids'].squeeze()
    attention_mask = inputs['attention_mask'].squeeze()
    return input_ids, attention_mask

def preprocess_data(question, answer, label, tokenizer):

    input_text = f"{question} [SEP] {answer}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids'].squeeze()
    attention_mask = inputs['attention_mask'].squeeze()
    return input_ids, attention_mask, torch.tensor(label)

def create_test_dataloader(data, tokenizer):
    
    # Apply preprocessing to your dataset
    X_data, attention_masks = zip(*[preprocess_test_data(question, answer, tokenizer) for question, answer in zip(data['question'], data['answer'])])

    # Pad input tensors to the same length
    X_data_padded = torch.nn.utils.rnn.pad_sequence(X_data, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)  

    # Convert the dataset to PyTorch tensors
    dataset = TensorDataset(X_data_padded, attention_masks_padded)
    
    return dataset

def create_dataloader(data, tokenizer):
    # Apply preprocessing to your dataset
    X_data, attention_masks, y_data = zip(*[preprocess_data(question, answer, label, tokenizer) for question, answer, label in zip(data['question'], data['answer'], data['label'])])

    # Pad input tensors to the same length
    X_data_padded = torch.nn.utils.rnn.pad_sequence(X_data, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0) 

    # Convert the dataset to PyTorch tensors
    dataset = TensorDataset(X_data_padded, attention_masks_padded, torch.stack(y_data))
    
    return dataset

def evaluate_test(model, test_dataloader, threshold, device):

    print('*' * 50)
    print('\n GENERATING PREDICTIONS FOR TEST...\n ')
    print('*' * 50)  

    # Inference loop on development set
    model.eval()
    test_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            test_predictions.extend(probabilities[:, 1].detach().cpu().numpy()) 

    # Convert predicted probabilities to predicted labels using a threshold
    test_predicted_labels = [1 if prob >= threshold else 0 for prob in test_predictions]
    
    count0 = 0
    count1 = 1
    
    for pred in test_predicted_labels:
        print(pred)
        if pred == 0:
            count0 += 1
        else:
            count1 += 1
    
    print('*' * 50)
    print(f'\nTOTAL PREDICTIONS: {len(test_predicted_labels)}\n')
    print(f'\nNO. OF LABELS = 1 : {count1}\n')
    print(f'\nNO. OF LABELS = 0 : {count0}\n')
    print('*' * 50)  
    
    return test_predicted_labels


def evaluate_dev(model, dev_dataloader, threshold, device):

    print('*' * 50)
    print('\n EVALUATING DEV\n ')
    print('*' * 50)  

    # Inference loop on development set
    model.eval()
    dev_predictions = []
    dev_true_labels = []

    with torch.no_grad():
        for batch in dev_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            dev_predictions.extend(probabilities[:, 1].detach().cpu().numpy()) 
            dev_true_labels.extend(labels.cpu().numpy())

    # Convert predicted probabilities to predicted labels using a threshold
    dev_predicted_labels = [1 if prob >= threshold else 0 for prob in dev_predictions]

    # Calculate and print confusion matrix
    cm = confusion_matrix(dev_true_labels, dev_predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    # Calculate and print classification report
    cr = classification_report(dev_true_labels, dev_predicted_labels)
    print("Classification Report:")
    print(cr)
    
    print('*' * 50)
    print('\nEVALUATION ENDED\n')
    print('*' * 50)  
    
    return dev_predicted_labels


def test(args):
    
    models = dict({
        1 : 'bert-base-uncased',
        2 : 'nlpaueb/legal-bert-base-uncased',
    })
    
    pretrained_model = models[int(args.pretrained_model)]        
    load_checkpoint_dir = args.load_checkpoint_dir
    gpu = args.gpu
    threshold = args.classifier_prob_threshold
    
    if args.dataset not in ['dev', 'test']:
        print("Only accepted values for dataset argument = ['test', 'dev']")
    
    test_data = pd.read_csv(f'data/semeval_data/{args.dataset}.csv')
    print(f'Shape of {args.dataset} dataset = {test_data.shape}')
    
    model, tokenizer = init(pretrained_model, load_checkpoint_dir)
    
    # Set up GPU if available
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    model.to(device)
    
    if args.dataset == 'dev':
        # Create data loaders
        test_dataset = create_dataloader(test_data, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)        
        y_test_pred = evaluate_dev(model, test_dataloader, threshold, device)
    else:
        # Create data loaders
        test_dataset = create_test_dataloader(test_data, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)            
        y_test_pred = evaluate_test(model, test_dataloader, threshold, device)
    
    if args.create_submission:
        create_submission(y_test_pred, args.dataset)
    
    # Clear cache before exiting
    with torch.device(f'cuda:{gpu}'):
        torch.cuda.empty_cache()
    
def create_submission(pred, dataset):
    
    
    evaluation_dataset = f'data/semeval_data/{dataset}.csv'
    df_eval_data = pd.read_csv(evaluation_dataset, index_col=0)

    df_result = pd.DataFrame(index=df_eval_data.index)
    df_result['result'] = pred
    
    print(df_result)
    
    df_result.to_csv('submission.csv')
    zip = zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED)
    zip.write('submission.csv')
    zip.close()
    
    print('Submission file created - submission.zip')
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-tuning pretrained models and evaluation for Legal Argument Reasoning task.')
    
    parser.add_argument('--pretrained_model', help='Integer specifying the huggingface pretrained model to load. One of \n1. BERT (bert-base-uncased), \n2. LegalBert (nlpaueb/legal-bert-base-uncased)', type=int, required=True)
    
    parser.add_argument('--load_checkpoint_dir', help='Path to checkpoint directory from which training checkpoint should be loaded.', required=False)
    
    parser.add_argument('--gpu', help='GPU to use', default=0, required=False, type=int)
    parser.add_argument('--classifier_prob_threshold', help='Probability threshold that separates the two classes', default=0.5, required=False, type=float)
        
    parser.add_argument('--create_submission', help="1 if creating submission file for competition, 0 otherwise. Default is 0.", default=0, type=int)
    
    parser.add_argument('--dataset', help="dev or test data", default='dev')
    
    args = parser.parse_args()
    
    test(args)
    