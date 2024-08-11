from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import pandas as pd
import torch

# Load the prepared dataset
df = pd.read_csv('prepared_dataset.csv')
dataset = Dataset.from_pandas(df)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['Answer'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets

# Load the pre-trained model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure pad_token_id is defined (otherwise it may result in an error)
model.config.pad_token_id = tokenizer.pad_token_id

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,  # Ensures that only the loss is calculated and stored
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
