#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import glob
import logging
import random
from itertools import groupby
from typing import Dict, List, Any
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

###############################################################################
# CONFIGURATION
###############################################################################

CLAUDE_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"
CLAUDE_MODEL = "claude-3-5-haiku-latest"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus", "split")
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results_ae", "Online", CLAUDE_MODEL)
PROMPTS_FILE = os.path.join(os.path.dirname(CORPUS_DIR), "prompts.csv")

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

###############################################################################
# LOGGING
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    prompt_map = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                prompt_map[row["language"].strip()] = row["prompt"].strip()
        logging.info(f"Loaded prompt prefixes from {file_path}.")
    except Exception as e:
        logging.error(f"Error loading prompt file {file_path}: {e}")
    return prompt_map

def ensure_dir_exists(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_json_list_field(json_str: str) -> List[Dict[str, Any]]:
    if not json_str:
        return []
    try:
        data = json.loads(json_str)
        return data if isinstance(data, list) else []
    except Exception:
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
            key = (int(row.get("Run_Number", "0")), row.get("Updated_Entry_ID", ""), 
                   row.get("Entry_ID", ""), row.get("Language", ""))
            done.add(key)
    return done

###############################################################################
# ONLINE INFERENCE CLASS
###############################################################################

class ClaudeLLM:
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> Dict[str, Any]:
        max_retries = 50
        for attempt in range(max_retries):
            start = time.time()
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                elapsed = time.time() - start
                text = message.content[0].text if message.content else ""
                return {"text": text, "raw": message.model_dump(), "time_taken": elapsed}
            except Exception as e:
                elapsed = time.time() - start
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(2)
        logging.error("Max retries reached.")
        sys.exit(1)

###############################################################################
# CSV PROCESSING
###############################################################################

def process_csv_file(csv_file: str, engine: ClaudeLLM, run_number: int, subprompts: Dict[str, str]):
    rel_path = os.path.relpath(csv_file, CORPUS_DIR)
    result_csv = os.path.join(RESULTS_BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    done_set = load_existing_results(result_csv)
    write_header = not os.path.isfile(result_csv)

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    
    manual_items = [row for row in rows if row.get("Manual_Include", "").strip().upper() == "Y"]
    non_manual = [row for row in rows if row.get("Manual_Include", "").strip().upper() != "Y"]
    selection = manual_items.copy()
    if len(selection) < 20:
        selection.extend(random.sample(non_manual, min(20 - len(selection), len(non_manual))))

    tasks = []
    for row in selection:
        updated_entry_id = row.get("Updated_Entry_ID", "")
        entry_id = row.get("Entry_ID", "")
        english_statement = row.get("Entry", "").strip()
        country_field = row.get("Country", "").strip("[]' ")
        master_tag = row.get("Master_Tag", "")
        translation_map = build_translation_map(row.get("humantranslations", ""), row.get("machinetranslations", ""))
        
        for json_lang, subprompt_key in LANG_PROMPT_MAP.items():
            key = (run_number, updated_entry_id, entry_id, json_lang)
            if key in done_set:
                continue
            statement_lang_text = english_statement if json_lang == "English" else translation_map.get(json_lang, "")
            if not statement_lang_text:
                continue
            final_prompt = subprompts.get(subprompt_key, "") + statement_lang_text
            tasks.append((key, updated_entry_id, entry_id, english_statement, statement_lang_text, country_field, master_tag, final_prompt, json_lang))

    results_concurrent = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {executor.submit(engine.generate, task[7]): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results_concurrent.append((task, result))

    with open(result_csv, "a", encoding="utf-8", newline="") as fout:
        fieldnames = ["Run_Number", "Updated_Entry_ID", "Entry_ID", "Statement_English", "Statement_Language",
                      "Time_Taken", "Reply", "API_Response", "Language", "Master_Tag", "Country"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        for task, result_dict in results_concurrent:
            writer.writerow({
                "Run_Number": run_number,
                "Updated_Entry_ID": task[1],
                "Entry_ID": task[2],
                "Statement_English": task[3],
                "Statement_Language": task[4],
                "Time_Taken": result_dict.get("time_taken", ""),
                "Reply": result_dict["text"],
                "API_Response": json.dumps(result_dict["raw"], ensure_ascii=False),
                "Language": task[8],
                "Master_Tag": task[6],
                "Country": task[5],
            })
            done_set.add(task[0])
    logging.info(f"Finished processing {csv_file}")

###############################################################################
# MAIN
###############################################################################

def main():
    logging.info("Starting Claude online inference...")
    engine = ClaudeLLM(api_key=CLAUDE_API_KEY, model=CLAUDE_MODEL)
    subprompts = load_prompts_from_file(PROMPTS_FILE)
    csv_files = glob.glob(os.path.join(CORPUS_DIR, "**/*.csv"), recursive=True)

    for run_number in range(1, 2):
        logging.info(f"Starting run {run_number}")
        for csv_file in csv_files:
            if "EMPTY" in os.path.basename(csv_file):
                continue
            process_csv_file(csv_file, engine, run_number, subprompts)
    logging.info("All processing completed.")

if __name__ == "__main__":
    main()

