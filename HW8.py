import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    problem_type="multi_label_classification",
    num_labels=22373
)
model.to(device)

# Load training data
with open("training-set-100000.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)["articles"]

train_texts = [article["title"] + " " + article["abstractText"] for article in train_data]
train_labels = [article["meshMajor"] for article in train_data]

mlb = MultiLabelBinarizer()
train_labels_binarized = mlb.fit_transform(train_labels)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

class PubMedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PubMedDataset(train_encodings, train_labels_binarized)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train(resume_from_checkpoint=True)

# ------------------------------------------
# Test Set Evaluation (Task 2)
# ------------------------------------------
# ------------------------------------------
# Manual evaluation on test set using DataLoader
# ------------------------------------------

# Load and process test set
with open("test-set-20000-rev2.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)["documents"]

test_texts = [article["title"] + " " + article["abstractText"] for article in test_data]
test_labels = [article["meshMajor"] for article in test_data]
test_labels_binarized = mlb.transform(test_labels)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
test_dataset = PubMedDataset(test_encodings, test_labels_binarized)

test_loader = DataLoader(test_dataset, batch_size=2)  # adjust if needed

model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating test set"):
        labels = batch['labels']
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        logits = torch.sigmoid(outputs.logits).cpu().numpy()
        all_test_preds.extend(logits)
        all_test_labels.extend(labels.numpy())

# Compute metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

preds_bin = (np.array(all_test_preds) > 0.5).astype(int)
labels_bin = np.array(all_test_labels)

accuracy = accuracy_score(labels_bin, preds_bin)
precision = precision_score(labels_bin, preds_bin, average="micro", zero_division=0)
recall = recall_score(labels_bin, preds_bin, average="micro", zero_division=0)
f1 = f1_score(labels_bin, preds_bin, average="micro", zero_division=0)

print("\nTest Set Evaluation:")
print(f"Micro Accuracy:  {accuracy:.4f}")
print(f"Micro Precision: {precision:.4f}")
print(f"Micro Recall:    {recall:.4f}")
print(f"Micro F1 Score:  {f1:.4f}")


# ------------------------------------------
# Judge Set Prediction (Task 3)
# ------------------------------------------
with open("judge-set-10000-unannotated.json", "r", encoding="utf-8") as f:
    judge_data = json.load(f)["documents"]

judge_texts = [article["title"] + " " + article["abstractText"] for article in judge_data]
judge_encodings = tokenizer(judge_texts, truncation=True, padding=True, max_length=512)
dummy_labels = [[0]*train_labels_binarized.shape[1]] * len(judge_texts)
judge_dataset = PubMedDataset(judge_encodings, dummy_labels)

batch_size = 2  # Adjust based on available memory
dataloader = DataLoader(judge_dataset, batch_size=batch_size)

model.eval()
all_preds = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        logits = torch.sigmoid(outputs.logits).cpu().numpy()
        all_preds.extend(logits)

pred_labels = mlb.inverse_transform((np.array(all_preds) > 0.5).astype(int))

# Prepare and save output
judge_pmids = [article["pmid"] for article in judge_data]
output = {
    "documents": [
        {"pmid": int(pmid), "labels": list(labels)}
        for pmid, labels in zip(judge_pmids, pred_labels)
    ]
}

with open("judge_predictions.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nPredictions saved to judge_predictions.json")
