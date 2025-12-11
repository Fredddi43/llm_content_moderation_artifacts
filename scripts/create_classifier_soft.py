import os
os.environ["TRANSFORMERS_CACHE"] = "hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict


# create dataset from file
def create_dataset(path_train="../DeBERTa training set/Train_Data_30k.xlsx", path_test="../DeBERTa training set/Test_Data_30k.xlsx"):

    # train data
    df = pd.read_excel(path_train, sheet_name="Sheet1")
    df = df.rename(columns={"Text": "text", "Label": "labels", "Train": "train"})
    ds_train = Dataset.from_pandas(df)

    # test data
    df = pd.read_excel(path_test, sheet_name="Sheet1")
    df = df.rename(columns={"Text": "text", "Label": "labels", "Test": "test"})
    ds_test = Dataset.from_pandas(df)

    ds = DatasetDict()

    ds['train'] = ds_train
    ds['test'] = ds_test

    return ds

# train the model
def create_classifier():

    print("logging in to huggingface")

    # input valid huggingface token
    login(token="...")

    # load dataset
    print("loading dataset")
    ds = create_dataset()

    # load tokenizer
    print("loading tokenizer")
    #tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_ds = ds.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = {0: "uncensored", 1: "censored"}
    label2id = {"uncensored": 0, "censored": 1}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large", num_labels=2, id2label=id2label, label2id=label2id, cache_dir="hf_cache"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = "Classifier_30k"

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-6,

        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=2,
        gradient_checkpointing=True,
        fp16=True,

#        per_device_train_batch_size=16,  # set to 16
#        per_device_eval_batch_size=16,  # set to 16
        num_train_epochs=50,  # set to 4
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
        #remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("training")
    trainer.train()
    trainer.push_to_hub()

create_classifier()

