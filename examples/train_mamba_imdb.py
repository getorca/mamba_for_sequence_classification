from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
import torch
from hf_mamba_classification_2 import MambaForSequenceClassification
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset


model_path = 'state-spaces/mamba-2.8b-hf'

tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# load and process the dataset
imdb = load_dataset("imdb") #['train'].train_test_split(train_size=200, test_size=50)

def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
tokenized_imdb = tokenized_imdb.remove_columns('attention_mask')
train_dataset = tokenized_imdb["train"].select(range(5000))
eval_dataset = tokenized_imdb["test"].select(range(1000))


# setup some evaluation metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = MambaForSequenceClassification.from_pretrained(
    model_path, 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id,
    use_cache=False  # This needs to be passed when using eval and training Mamba for sequence classification otherwise it will raise an error
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir="mamba_imdb_classification",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    # eval_steps=1000,
    save_strategy="epoch",
    # load_best_model_at_end=True,
    lr_scheduler_type="cosine",
    optim='paged_adamw_8bit',
    # push_to_hub=True,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

    
)

trainer.train()

