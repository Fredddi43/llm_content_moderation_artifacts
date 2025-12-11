#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import glob
import logging
import sys
import random
from typing import Dict, List, Any

from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

sys.setrecursionlimit(10000)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

###############################################################################
# CONFIGURATION
###############################################################################

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "google--gemma-3-27b-it")
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "split")
RESULTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "results_ae", "google--gemma-3-27b-it")

PROMPTS_FILE = os.path.join(os.path.dirname(CORPUS_DIR), "prompts.csv")
LOCATION_PROMPTS_PATTERN = os.path.join(os.path.dirname(CORPUS_DIR), "prompts_with location_info_*.csv")

LANG_PROMPT_MAP = {
    "English": "English",
    "Arabic": "Arabic",
    "Chinese (Simplified)": "Chinese_Simplified",
    "Chinese (Traditional)": "Chinese_Traditional",
    "Spanish": "Spanish",
    "German": "German",
    "Portuguese": "Portuguese",
    "Russian": "Russian",
    "Turkish": "Turkish",
    "Hindi": "Hindi",
    "Afrikaans": "Afrikaans",
    "Zulu": "Zulu",
    "Persian (Farsi)": "Persian",
}

TARGET_NUM_ITEMS = 20

###############################################################################
# PROMPT LOADING
###############################################################################

def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    prompt_map = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                lang = row["language"].strip()
                prompt_text = row["prompt"].strip()
                prompt_map[lang] = prompt_text
        logging.info(f"Loaded prompt prefixes from {file_path}.")
    except Exception as e:
        logging.error(f"Error loading prompt file {file_path}: {e}")
    return prompt_map

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def ensure_dir_exists(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_json_list_field(json_str: str) -> List[Dict[str, Any]]:
    if not json_str:
        return []
    try:
        data = json.loads(json_str)
        return data if isinstance(data, list) else []
    except Exception as e:
        logging.debug(f"JSON parsing error: {e}")
        return []

def build_translation_map(human_str: str, machine_str: str) -> Dict[str, str]:
    human_data = load_json_list_field(human_str)
    machine_data = load_json_list_field(machine_str)
    human_map = {item.get("language", "").strip(): item.get("translation", "").strip()
                 for item in human_data if item.get("language") and item.get("translation")}
    machine_map = {item.get("language", "").strip(): item.get("translation", "").strip()
                   for item in machine_data if item.get("language") and item.get("translation")}
    combined_map = {}
    for lang in set(human_map.keys()).union(machine_map.keys()):
        combined_map[lang] = human_map.get(lang) or machine_map.get(lang, "")
    return combined_map

def load_existing_results(result_file: str) -> set:
    if not os.path.isfile(result_file):
        return set()
    done = set()
    with open(result_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("Updated_Entry_ID", ""), row.get("Entry_ID", ""), row.get("Language", ""))
            done.add(key)
    return done

def select_items_from_csv(csv_file: str, target: int = TARGET_NUM_ITEMS) -> List[Dict[str, Any]]:
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))
    manual = [row for row in reader if row.get("Manual_Include", "").strip().upper() == "Y"]
    non_manual = [row for row in reader if row.get("Manual_Include", "").strip().upper() != "Y"]
    if len(manual) >= target:
        return manual
    else:
        remaining = target - len(manual)
        additional = random.sample(non_manual, min(remaining, len(non_manual))) if non_manual else []
        return manual + additional

###############################################################################
# MAIN INFERENCE LOGIC (Using Transformers)
###############################################################################

def run_inference(prompts: List[str], model, tokenizer) -> List[str]:
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9
        )
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(result_text)
    return results

###############################################################################
# CSV PROCESSING
###############################################################################

def process_csv_file(csv_file: str, model, tokenizer, results_base_dir: str, subprompts: Dict[str, str]):
    rel_path = os.path.relpath(csv_file, CORPUS_DIR)
    result_csv = os.path.join(results_base_dir, rel_path)
    ensure_dir_exists(result_csv)
    done_set = load_existing_results(result_csv)
    if len(done_set) >= TARGET_NUM_ITEMS:
        logging.info(f"Skipping CSV {csv_file} because {len(done_set)} items have already been processed.")
        return

    selected_rows = select_items_from_csv(csv_file, target=TARGET_NUM_ITEMS)
    logging.info(f"Selected {len(selected_rows)} items from {csv_file} for processing.")

    write_header = not os.path.isfile(result_csv)
    with open(result_csv, "a", encoding="utf-8", newline="") as fout:
        fieldnames = ["Updated_Entry_ID", "Entry_ID", "Statement_English", "Statement_Language",
                      "Time_Taken", "Reply", "Language", "Master_Tag", "Country"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()

        for row in selected_rows:
            updated_entry_id = row.get("Updated_Entry_ID", "")
            entry_id = row.get("Entry_ID", "")
            english_statement = row.get("Entry", "").strip()
            if not english_statement:
                english_statement = row.get("Statement_English", "").strip()
            country = row.get("Country", "").strip("[]' ")
            master_tag = row.get("Master_Tag", "")
            human_str = row.get("humantranslations", "")
            machine_str = row.get("machinetranslations", "")
            translation_map = build_translation_map(human_str, machine_str)
            for json_lang, subprompt_key in LANG_PROMPT_MAP.items():
                key = (updated_entry_id, entry_id, json_lang)
                if key in done_set:
                    continue
                if json_lang == "English":
                    statement_lang_text = english_statement
                else:
                    statement_lang_text = translation_map.get(json_lang, "")
                    if not statement_lang_text:
                        logging.debug(f"No translation found for {json_lang} in Entry_ID {entry_id}; skipping.")
                        continue
                prefix = subprompts.get(subprompt_key, "")
                final_prompt = prefix + statement_lang_text
                start_time = time.time()
                outputs = run_inference([final_prompt], model, tokenizer)
                elapsed = time.time() - start_time
                reply_text = outputs[0] if outputs else ""
                writer.writerow({
                    "Updated_Entry_ID": updated_entry_id,
                    "Entry_ID": entry_id,
                    "Statement_English": english_statement,
                    "Statement_Language": statement_lang_text,
                    "Time_Taken": f"{elapsed:.4f}",
                    "Reply": reply_text,
                    "Language": json_lang,
                    "Master_Tag": master_tag,
                    "Country": country,
                })
                fout.flush()
                done_set.add(key)
                logging.debug(f"Processed prompt for {json_lang} in Entry_ID {entry_id} in {elapsed:.4f} seconds.")
    logging.info(f"Finished processing CSV: {csv_file}")

def run_experiment(prompt_file: str, results_subfolder: str, model, tokenizer):
    subprompts = load_prompts_from_file(prompt_file)
    run_results_dir = os.path.join(RESULTS_BASE_DIR, results_subfolder)
    logging.info(f"Experiment using {prompt_file} will save results in {run_results_dir}")
    csv_files = glob.glob(os.path.join(CORPUS_DIR, "**/*.csv"), recursive=True)
    for csv_file in csv_files:
        if "EMPTY" in os.path.basename(csv_file):
            logging.info(f"Skipping file with 'EMPTY': {csv_file}")
            continue
        logging.info(f"Processing CSV file: {csv_file}")
        process_csv_file(csv_file, model, tokenizer, run_results_dir, subprompts)

def main():
    logging.info("Starting main processing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        trust_remote_code=True
    )
    
    logging.info("Starting experiment using prompts.csv")
    run_experiment(PROMPTS_FILE, "run_general", model, tokenizer)
    
    location_files = glob.glob(LOCATION_PROMPTS_PATTERN)
    for loc_file in location_files:
        country_name = os.path.basename(loc_file)
        if country_name.startswith("prompts_with_location_info_") and country_name.endswith(".csv"):
            country = country_name[len("prompts_with_location_info_"):-len(".csv")]
        else:
            country = country_name
        logging.info(f"Starting experiment for location {country}")
        run_experiment(loc_file, f"run_{country}", model, tokenizer)
    logging.info("All experiments processed.")

if __name__ == "__main__":
    main()

