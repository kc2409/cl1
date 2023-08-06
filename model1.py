import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

# Assuming you have the 'transformers' library installed, if not, install it using: pip install transformers

labelled_data_path='hc_train.csv'
unlabelled_data_path='hc_test.csv'

labeled_df=pd.read_csv(labelled_data_path)
unlabeled_df=pd.read_csv(unlabelled_data_path)

labeled_data=labeled_df[['content', 'target', 'stance_label']].to_dict('records')
unlabeled_data = unlabeled_df[['content', 'target']].to_dict('records')

#print("Labeled Data:")
#for data_item in labeled_data:
 #   print(data_item, type(data_item['target']), type(data_item['content']))

#print("Unlabeled Data:")
#for data_item in unlabeled_data:
 #   print(data_item, type(data_item['target']), type(data_item['content']))

class StanceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, labeled=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labeled = labeled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = self.data[idx]['content']
        target = self.data[idx]['target']

        if not isinstance(content, str) or not isinstance(target, str):
            if not isinstance(content, str):
                content = str(content)  
            if not isinstance(target, str):
                target = str(target) 

        inputs = self.tokenizer.encode_plus(
            target,
            content,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        if self.labeled:
            stance_label = torch.tensor(self.data[idx]['stance_label'], dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'stance_label': stance_label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_length = 128
labeled_dataset = StanceDataset(labeled_data, tokenizer, max_length, labeled=True)
unlabeled_dataset = StanceDataset(unlabeled_data, tokenizer, max_length, labeled=False)

# Create DataLoaders
batch_size = 8
labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

class StanceDetectionModel(nn.Module):
    def __init__(self, num_labels):
        super(StanceDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# Create the Stance Detection Model
num_labels = 3  # Replace with the actual number of stance labels in your data
model = StanceDetectionModel(num_labels)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        stance_labels = None
        if 'stance_label' in batch:
            stance_labels = batch['stance_label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #logits = 
        print(type(outputs))

        if stance_labels is not None:
            loss = criterion(outputs, stance_labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    return total_loss / len(dataloader)


# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    labeled_loss = train_model(model, labeled_dataloader, optimizer, criterion, device)
    unlabeled_loss = train_model(model, unlabeled_dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch + 1} - Labeled Loss: {labeled_loss:.4f} - Unlabeled Loss: {unlabeled_loss:.4f}")
