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

import openai

###############################################################################
# CONFIGURATION
###############################################################################

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
ONLINE_MODEL = "gpt-4.1-mini"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus", "split")
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results_ae", "Online", ONLINE_MODEL)
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

SELECTIONS_FILE = os.path.join(RESULTS_BASE_DIR, f"selections_{ONLINE_MODEL}.json")

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
                lang = row["language"].strip()
                prompt_text = row["prompt"].strip()
                prompt_map[lang] = prompt_text
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
    except Exception as e:
        logging.info(f"JSON parsing error: {e}")
        return []

def build_translation_map(human_str: str, machine_str: str) -> Dict[str, str]:
    human_data = load_json_list_field(human_str)
    machine_data = load_json_list_field(machine_str)
    human_map = {
        item.get("language", "").strip(): item.get("translation", "").strip()
        for item in human_data
        if item.get("language", "").strip() and item.get("translation", "").strip()
    }
    machine_map = {
        item.get("language", "").strip(): item.get("translation", "").strip()
        for item in machine_data
        if item.get("language", "").strip() and item.get("translation", "").strip()
    }
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
            key = (
                int(row.get("Run_Number", "0")),
                row.get("Updated_Entry_ID", ""),
                row.get("Entry_ID", ""),
                row.get("Language", ""),
            )
            done.add(key)
    return done

###############################################################################
# SELECTION FUNCTIONS
###############################################################################

def load_all_selections() -> Dict[str, List[Dict[str, Any]]]:
    if os.path.exists(SELECTIONS_FILE):
        try:
            with open(SELECTIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error("Error loading selections file: %s", e)
            return {}
    return {}

def save_all_selections(all_selections: Dict[str, List[Dict[str, Any]]]):
    os.makedirs(os.path.dirname(SELECTIONS_FILE), exist_ok=True)
    try:
        with open(SELECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_selections, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error("Error saving selections file: %s", e)

def load_selection_for_csv(csv_file: str) -> List[Dict[str, Any]]:
    all_selections = load_all_selections()
    rel_path = os.path.relpath(csv_file, CORPUS_DIR)
    key = rel_path
    return all_selections.get(key, None)

def save_selection_for_csv(csv_file: str, selection: List[Dict[str, Any]]):
    all_selections = load_all_selections()
    rel_path = os.path.relpath(csv_file, CORPUS_DIR)
    key = rel_path
    all_selections[key] = selection
    save_all_selections(all_selections)

def select_items_from_csv(csv_file: str, num_items: int) -> List[Dict[str, Any]]:
    items = []
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    manual_items = [
        row for row in items if row.get("Manual_Include", "").strip().upper() == "Y"
    ]
    non_manual_items = [
        row for row in items if row.get("Manual_Include", "").strip().upper() != "Y"
    ]
    
    selection = manual_items.copy()
    remaining_needed = num_items - len(selection)
    if remaining_needed > 0 and non_manual_items:
        additional = random.sample(non_manual_items, min(remaining_needed, len(non_manual_items)))
        selection.extend(additional)
    return selection

###############################################################################
# ONLINE INFERENCE CLASS
###############################################################################

def chat_completion_create(model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    response = openai.responses.create(model=model, input=messages, timeout=30)
    return response.to_dict()

class OnlineLLM:
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        max_retries = 50
        backoff_times = [0.5, 1, 2]
        for attempt in range(max_retries):
            start = time.time()
            try:
                response = chat_completion_create(self.model, messages)
                end = time.time()
                time_taken = end - start
                if response.get("error") is not None:
                    err_str = json.dumps(response["error"], indent=2, ensure_ascii=False)
                    logging.error("API error encountered: %s", err_str)
                    raise Exception("API error: " + err_str)
                logging.info("Received API response in %.2f seconds", time_taken)
                text = (
                    response["output"][0]["content"][0]["text"].strip()
                    if response.get("output")
                    else ""
                )
                return {"text": text, "raw": response, "time_taken": time_taken}
            except Exception as e:
                end = time.time()
                time_taken = end - start
                err_str = str(e)
                if "429" in err_str:
                    backoff = backoff_times[min(attempt, len(backoff_times) - 1)]
                    logging.warning("Error 429 encountered. Backing off for %.2f seconds.", backoff)
                    time.sleep(backoff)
                else:
                    logging.warning("Encountered error (attempt %d/%d): %s", attempt + 1, max_retries, err_str)
                    time.sleep(2)
        logging.error("Max retries reached for prompt. Terminating script.")
        sys.exit(1)

def run_online_inference(prompts: List[str], engine: OnlineLLM) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_prompt = {executor.submit(engine.generate, prompt): prompt for prompt in prompts}
        for future in as_completed(future_to_prompt):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error("Error in concurrent API call: %s", e)
                raise
    return results

###############################################################################
# CSV PROCESSING
###############################################################################

def process_csv_file(
    csv_file: str,
    engine: OnlineLLM,
    run_number: int,
    subprompts: Dict[str, str],
):
    rel_path = os.path.relpath(csv_file, CORPUS_DIR)
    result_csv = os.path.join(RESULTS_BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    done_set = load_existing_results(result_csv)
    write_header = not os.path.isfile(result_csv)

    selection = load_selection_for_csv(csv_file)
    if selection is None:
        selection = select_items_from_csv(csv_file, 20)
        save_selection_for_csv(csv_file, selection)

    tasks = []
    for row in selection:
        updated_entry_id = row.get("Updated_Entry_ID", "")
        entry_id = row.get("Entry_ID", "")
        english_statement = row.get("Entry", "").strip()
        country_field = row.get("Country", "").strip("[]' ")
        master_tag = row.get("Master_Tag", "")
        human_str = row.get("humantranslations", "")
        machine_str = row.get("machinetranslations", "")
        translation_map = build_translation_map(human_str, machine_str)
        for json_lang, subprompt_key in LANG_PROMPT_MAP.items():
            key = (run_number, updated_entry_id, entry_id, json_lang)
            if key in done_set:
                continue
            if json_lang == "English":
                statement_lang_text = english_statement
            else:
                statement_lang_text = translation_map.get(json_lang, "")
                if not statement_lang_text:
                    logging.info(f"No translation for {json_lang} in Entry_ID {entry_id}; skipping.")
                    continue
            prefix = subprompts.get(subprompt_key, "")
            final_prompt = prefix + statement_lang_text
            tasks.append((key, updated_entry_id, entry_id, english_statement, statement_lang_text, country_field, master_tag, final_prompt, json_lang))

    results_concurrent = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {executor.submit(engine.generate, task[7]): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results_concurrent.append((task, result))
            except Exception as e:
                logging.error("Error in concurrent API call for task %s: %s", task, e)
                raise

    with open(result_csv, "a", encoding="utf-8", newline="") as fout:
        fieldnames = [
            "Run_Number", "Updated_Entry_ID", "Entry_ID", "Statement_English", "Statement_Language",
            "Time_Taken", "Reply", "API_Response", "Language", "Master_Tag", "Country",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        for task, result_dict in results_concurrent:
            (key, updated_entry_id, entry_id, english_statement, statement_lang_text, country_field, master_tag, final_prompt, json_lang) = task
            reply_text = result_dict["text"]
            time_taken = result_dict.get("time_taken", "")
            api_response_str = json.dumps(result_dict["raw"], ensure_ascii=False)
            writer.writerow({
                "Run_Number": run_number,
                "Updated_Entry_ID": updated_entry_id,
                "Entry_ID": entry_id,
                "Statement_English": english_statement,
                "Statement_Language": statement_lang_text,
                "Time_Taken": time_taken,
                "Reply": reply_text,
                "API_Response": api_response_str,
                "Language": json_lang,
                "Master_Tag": master_tag,
                "Country": country_field,
            })
            done_set.add(key)
        fout.flush()
    logging.info(f"Finished processing CSV: {csv_file} for run {run_number}. Total tasks processed: {len(results_concurrent)}")

###############################################################################
# MAIN EXECUTION LOGIC
###############################################################################

def main():
    logging.info("Starting online inference processing...")

    openai.api_key = OPENAI_API_KEY

    engine = OnlineLLM(model=ONLINE_MODEL)
    subprompts = load_prompts_from_file(PROMPTS_FILE)

    csv_files = glob.glob(os.path.join(CORPUS_DIR, "**/*.csv"), recursive=True)
    
    for run_number in range(1, 6):
        logging.info(f"Starting run {run_number}")
        for csv_file in csv_files:
            if "EMPTY" in os.path.basename(csv_file):
                logging.info(f"Skipping {csv_file} because it contains 'EMPTY'.")
                continue
            process_csv_file(csv_file, engine, run_number, subprompts)

    logging.info("All processing completed.")

if __name__ == "__main__":
    main()

