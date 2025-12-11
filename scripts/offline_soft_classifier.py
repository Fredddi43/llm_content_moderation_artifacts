#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import glob
import logging
import sys
import torch
import fcntl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

###############################################################################
# SIMPLE FILE LOCK
###############################################################################
class FileLock:
    def __init__(self, lockfile: str):
        self.lockfile = lockfile
        self.fd = None

    def __enter__(self):
        self.fd = open(self.lockfile, 'a+')
        fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            self.fd.close()
        try:
            os.remove(self.lockfile)
        except OSError:
            pass

###############################################################################
# CONFIGURATION
###############################################################################
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "classifier_30k")
RESULTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# map model output IDs to label strings
LABEL_MAP = {0: "uncensored", 1: "censored"}

# device and batch-size for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

###############################################################################
# SETUP LOGGING
###############################################################################
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

###############################################################################
# LOAD MODEL
###############################################################################
logging.info(f"Loading model from {MODEL_DIR} onto {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

###############################################################################
# HELPERS
###############################################################################
def classify_batch(texts):
    """Tokenize and run a batch of texts through the HF model."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [LABEL_MAP.get(p, "uncensored") for p in preds]

def process_csv_in_place(path: str):
    lock_path = path + ".lock"

    with FileLock(lock_path):
        logging.info(f"→ Locked and processing {path}")

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                logging.info(f"Empty CSV: {path}")
                return
            fieldnames = reader.fieldnames

        if "deberta_soft_label" not in fieldnames:
            fieldnames = list(fieldnames) + ["deberta_soft_label"]

        batch_texts = []
        batch_indices = []

        for idx, row in enumerate(rows):
            reply = row.get("EnglishTranslation", "").strip()
            existing_label = row.get("deberta_soft_label", "").strip()

            if reply and not existing_label:
                batch_texts.append(reply)
                batch_indices.append(idx)

        if not batch_texts:
            logging.info(f"No rows to process in {path}")
            return

        logging.info(f"Classifying {len(batch_texts)} rows in {path}")
        predictions = []
        
        for i in tqdm(range(0, len(batch_texts), BATCH_SIZE), desc="Batches", leave=False):
            batch = batch_texts[i : i + BATCH_SIZE]
            preds = classify_batch(batch)
            predictions.extend(preds)

        for idx_in_batch, row_idx in enumerate(batch_indices):
            label = predictions[idx_in_batch]
            if label == "censored":
                rows[row_idx]["deberta_soft_label"] = "moderated"
            else:
                rows[row_idx]["deberta_soft_label"] = "unmoderated"

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)

        logging.info(f"✓ Finished {path}")

###############################################################################
# MAIN
###############################################################################
def main():
    pattern = os.path.join(RESULTS_BASE_DIR, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    logging.info(f"Found {len(files)} CSV files to process")

    for path in tqdm(files, desc="Files", unit="file"):
        try:
            process_csv_in_place(path)
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            continue

    logging.info("All files processed.")

if __name__ == "__main__":
    main()

