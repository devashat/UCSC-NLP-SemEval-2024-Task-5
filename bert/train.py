import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import time
import os 
import argparse
from tqdm import tqdm


def init(pretrained_model, load_checkpoint_dir):

    if load_checkpoint_dir:
        model = AutoModelForSequenceClassification.from_pretrained(load_checkpoint_dir, torch_dtype="auto")
        print(f'Loading model from checkpoint: {load_checkpoint_dir}')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, torch_dtype="auto")
        print(f'Loading model from huggingface: {pretrained_model}')        
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    # Load dataset train and dev
    train_data = pd.read_csv('data/semeval_data/train.csv')
    dev_data = pd.read_csv('data/semeval_data/dev.csv')

    train_data.head()
    dev_data.head()

    print(f'Shape of train data = {train_data.shape}')
    print(f'Shape of dev data = {dev_data.shape}')
    
    return model, tokenizer, train_data, dev_data

# Example data preprocessing with [SEP] token and padding
def preprocess_data(question, answer, label, tokenizer):

    input_text = f"{question} [SEP] {answer}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids'].squeeze()
    attention_mask = inputs['attention_mask'].squeeze()
    return input_ids, attention_mask, torch.tensor(label)


def create_dataloader(data, tokenizer):
    
    # Apply preprocessing to your dataset
    X_data, attention_masks, y_data = zip(*[preprocess_data(question, answer, label, tokenizer) for question, answer, label in zip(data['question'], data['answer'], data['label'])])

    # Pad input tensors to the same length
    X_data_padded = torch.nn.utils.rnn.pad_sequence(X_data, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)  # Assuming attention masks use 0 for padding

    # Convert the dataset to PyTorch tensors
    dataset = TensorDataset(X_data_padded, attention_masks_padded, torch.stack(y_data))
    
    return dataset


# Function to save model and training hyperparameters
def save_model_and_params(model, optimizer, scheduler, epoch, loss, train_accuracy, train_f1, dev_accuracy, dev_f1, final_ckpt_dir):
    
    ckpt_dir = f'{final_ckpt_dir}/epoch{str(epoch).zfill(4)}'
    
    # Save model state
    model_state_dict = model.state_dict()

    # Save model configuration
    model_config = model.config
    model_config.save_pretrained(ckpt_dir)

    # Save model weights
    model.save_pretrained(ckpt_dir)

    # Save optimizer and scheduler state
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'train_accuracy': train_accuracy,
        'dev_accuracy': dev_accuracy,
        'train_f1': train_f1,
        'dev_f1': dev_f1
    }
    # with open(os.path.join(ckpt_dir, 'parameters.json'), 'w') as fp:
    #     json.dump(checkpoint, fp, indent=4)
    checkpoint_path = os.path.join(ckpt_dir, f'ckpt-epoch-{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)

def train(model, train_dataloader, dev_dataloader, gpu, threshold, device, checkpoint_dir, num_epochs, learning_rate, pretrained_model):

    print('*' * 50)
    print('\nTRAINING STARTED\n')
    print('*' * 50)    

    # Get current time for saving checkpoint folder
    current_time_struct = time.localtime()
    formatted_date = time.strftime("%Y%m%d", current_time_struct)
    timestamp = int(time.time())
    curr_time = f"{formatted_date}_{timestamp}"
    
    model_name = pretrained_model.split('/')[-1]
    final_ckpt_dir = os.path.join(checkpoint_dir, f'{curr_time}_{model_name}')
    if not os.path.exists(final_ckpt_dir):
        os.makedirs(final_ckpt_dir)

    model.to(device)

    # Set up training parameters
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=5e5)

    # Create Tensorboard writer
    writer = SummaryWriter(log_dir=f'runs/{curr_time}_{model_name}')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Lists to store predicted labels and true labels for training set
        train_predictions = []
        train_labels = []
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        
        for batch in progress_bar:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Append predicted labels and true labels for training set
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            # Assuming binary classification and extracting probability for the positive class
            train_predictions.extend(probabilities[:, 1].detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'Loss': total_loss / (progress_bar.n + 1)})

        average_loss = total_loss / len(train_dataloader)

        # Calculate and print training accuracy
        train_predicted_labels = [1 if prob >= threshold else 0 for prob in train_predictions]
        train_accuracy = accuracy_score(train_labels, train_predicted_labels)
        train_f1 = f1_score(train_labels, train_predicted_labels, average='macro')

        # Write to tensorboard
        writer.add_scalar('Train/Loss', average_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_accuracy, epoch)    
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Training Accuracy: {train_accuracy * 100:.2f}%")

        # Inference loop on development set
        model.eval()
        dev_predictions = []
        dev_labels = []

        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                dev_predictions.extend(probabilities[:, 1].detach().cpu().numpy())
                dev_labels.extend(labels.cpu().numpy())

        # Calculate and print development accuracy and F1 score
        dev_predicted_labels = [1 if prob >= threshold else 0 for prob in dev_predictions]
        dev_accuracy = accuracy_score(dev_labels, dev_predicted_labels)
        dev_f1 = f1_score(dev_labels, dev_predicted_labels, average='macro')
        
        writer.add_scalar('Dev/Accuracy', dev_accuracy, epoch)
        writer.add_scalar('Dev/F1', dev_f1, epoch)

        # Save model checkpoint every 5 epochs 
        if (epoch+1) % 5 == 0 or epoch == num_epochs-1 :
            save_model_and_params(model, optimizer, scheduler, epoch, loss, train_accuracy, train_f1, dev_accuracy, dev_f1, final_ckpt_dir)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Development Accuracy: {dev_accuracy * 100:.2f}%, Development F1 Score: {dev_f1 * 100:.2f}%")


    print('*' * 50)
    print('\nTRAINING ENDED\n')
    print('*' * 50)  


def evaluate(model, dev_dataloader, dev_data, threshold, device):

    print('*' * 50)
    print('\n EVALUATION STARTED\n ')
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
            dev_predictions.extend(probabilities[:, 1].detach().cpu().numpy())  # Detach gradients before converting to NumPy
            dev_true_labels.extend(labels.cpu().numpy())

    # Convert predicted probabilities to predicted labels using a threshold (e.g., 0.5 for binary classification)
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


def main(args):
            
    models = dict({
        1 : 'bert-base-uncased',
        2 : 'nlpaueb/legal-bert-base-uncased',
    })
    
    pretrained_model = models[int(args.pretrained_model)]        
    save_checkpoint_dir = args.save_checkpoint_dir
    load_checkpoint_dir = args.load_checkpoint_dir
    gpu = args.gpu
    num_epochs = args.num_epochs
    threshold = args.classifier_prob_threshold
    learning_rate = args.learning_rate
    
    model, tokenizer, train_data, dev_data = init(pretrained_model, load_checkpoint_dir)
    
    # Create data loaders
    train_dataset = create_dataloader(train_data, tokenizer)
    dev_dataset = create_dataloader(dev_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    
    # Set up GPU if available
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    train(model, train_dataloader, dev_dataloader, gpu, threshold, device, save_checkpoint_dir, num_epochs, learning_rate, pretrained_model)
    
    evaluate(model, dev_dataloader, dev_data, threshold, device)
        
    # Clear cache before exiting
    with torch.device(f'cuda:{gpu}'):
        torch.cuda.empty_cache()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-tuning pretrained models and evaluation for Legal Argument Reasoning task.')
    
    parser.add_argument('--pretrained_model', help='Integer specifying the huggingface pretrained model to load. One of \n1. BERT (bert-base-uncased), \n2. LegalBERT (nlpaueb/legal-bert-base-uncased)', type=int, required=True)
    
    parser.add_argument('--save_checkpoint_dir', help='Path to directory where training checkpoints should be stored', default='/home/srajeshj/nlp243/project/LegalArgumentReasoning/checkpoints', required=False)
    parser.add_argument('--load_checkpoint_dir', help='Path to checkpoint directory from which training checkpoint should be loaded.', required=False)
    
    parser.add_argument('--gpu', help='GPU to use', default=0, required=False, type=int)
    parser.add_argument('--num_epochs', help='Number of epochs', default=100, required=False, type=int)
    parser.add_argument('--classifier_prob_threshold', help='Probability threshold that separates the two classes', default=0.5, required=False, type=float)
    parser.add_argument('--learning_rate', help='Learning Rate for training the model', default=2e-5, required=False, type=float)
            
    args = parser.parse_args()
    
    main(args)