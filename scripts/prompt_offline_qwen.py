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

from vllm import LLM, SamplingParams

sys.setrecursionlimit(10000)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

###############################################################################
# CONFIGURATION
###############################################################################

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen--Qwen2.5-72B-Instruct")
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "split")
RESULTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "results_ae", "Qwen2.5-72B-Instruct")

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

###############################################################################
# MAIN INFERENCE LOGIC (Using vLLM)
###############################################################################

def run_inference(prompts: List[str], engine: LLM) -> List[str]:
    sampling_params = SamplingParams(max_tokens=1000, temperature=0.7, top_p=0.9)
    outputs = engine.generate(prompts, sampling_params)
    results = []
    for output in outputs:
        if output.outputs:
            results.append(output.outputs[0].text.strip())
        else:
            results.append("")
    return results

###############################################################################
# CSV PROCESSING
###############################################################################

def process_csv_file(csv_file: str, engine: LLM, base_corpus_dir: str, results_base_dir: str, subprompts: Dict[str, str]):
    rel_path = os.path.relpath(csv_file, base_corpus_dir)
    result_csv = os.path.join(results_base_dir, rel_path)
    ensure_dir_exists(result_csv)
    done_set = load_existing_results(result_csv)
    write_header = not os.path.isfile(result_csv)
    logging.info(f"Starting processing file: {csv_file}")
    
    with open(csv_file, "r", encoding="utf-8", newline="") as fin:
        rows = list(csv.DictReader(fin))
    
    manual_rows = [row for row in rows if row.get("Manual_Include", "").strip().upper() == "Y"]
    non_manual_rows = [row for row in rows if row.get("Manual_Include", "").strip().upper() != "Y"]
    selected_rows = manual_rows.copy()
    if len(selected_rows) < TARGET_NUM_ITEMS:
        remaining = TARGET_NUM_ITEMS - len(selected_rows)
        if non_manual_rows:
            selected_rows.extend(random.sample(non_manual_rows, min(remaining, len(non_manual_rows))))
    logging.info(f"Selected {len(selected_rows)} items from {csv_file} for processing.")

    with open(result_csv, "a", encoding="utf-8", newline="") as fout:
        fieldnames = ["Updated_Entry_ID", "Entry_ID", "Statement_English", "Statement_Language", "Time_Taken", "Reply", "Language", "Master_Tag", "Country"]
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
                        logging.debug(f"No translation for {json_lang} in Entry_ID {entry_id}; skipping.")
                        continue
                prefix = subprompts.get(subprompt_key, "")
                final_prompt = prefix + statement_lang_text
                start_time = time.time()
                outputs = run_inference([final_prompt], engine=engine)
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
    logging.info(f"Finished processing file: {csv_file}")

def run_experiment(prompt_file: str, run_subfolder: str, engine: LLM):
    subprompts = load_prompts_from_file(prompt_file)
    run_results_dir = os.path.join(RESULTS_BASE_DIR, run_subfolder)
    logging.info(f"Experiment using {prompt_file} will save results in {run_results_dir}")
    csv_files = glob.glob(os.path.join(CORPUS_DIR, "**/*.csv"), recursive=True)
    for csv_file in csv_files:
        if "EMPTY" in os.path.basename(csv_file):
            logging.info(f"Skipping file with 'EMPTY': {csv_file}")
            continue
        logging.info(f"Processing CSV file: {csv_file}")
        process_csv_file(csv_file, engine, CORPUS_DIR, run_results_dir, subprompts)

def main():
    logging.info("Starting main processing...")
    engine = LLM(model=MODEL_DIR, enforce_eager=True, trust_remote_code=True, tensor_parallel_size=2, max_model_len=10000, gpu_memory_utilization=0.95)
    
    for run_number in range(1, 4):
        run_folder = f"run{run_number}"
        logging.info(f"Starting general experiment run: {run_folder}")
        run_experiment(PROMPTS_FILE, run_folder, engine)
        
        location_files = glob.glob(LOCATION_PROMPTS_PATTERN)
        for loc_file in location_files:
            country_name = os.path.basename(loc_file)
            if country_name.startswith("prompts_with_location_info_") and country_name.endswith(".csv"):
                country = country_name[len("prompts_with_location_info_"):-len(".csv")]
            else:
                country = country_name
            run_folder = f"location_run_{country}_{run_number}"
            logging.info(f"Starting location experiment for {country} run: {run_folder}")
            run_experiment(loc_file, run_folder, engine)
    logging.info("All experiments processed.")

if __name__ == "__main__":
    main()

