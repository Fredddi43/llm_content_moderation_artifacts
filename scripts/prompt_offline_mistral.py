#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# disable v1 async engine so we use the V0 AsyncLLMEngine API
os.environ["VLLM_USE_V1"] = "0"

import csv
import json
import time
import glob
import logging
import sys
import random
import uuid
import asyncio
from typing import Dict, List, Any

import chardet

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

###############################################################################
# CONFIGURATION
###############################################################################

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "mistralai--Mistral-Small-3.1-24B-Instruct-2503")
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "split")
RESULTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "results_ae", "mistralai--Mistral-Small-3.1-24B-Instruct-2503")

PROMPTS_FILE = os.path.join(os.path.dirname(CORPUS_DIR), "prompts.csv")
LOCATION_PROMPTS_PATTERN = os.path.join(os.path.dirname(CORPUS_DIR), "prompts_with_location_info_*.csv")

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
# UTILITY FUNCTIONS
###############################################################################

def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    prompt_map: Dict[str, str] = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                prompt_map[row["language"].strip()] = row["prompt"].strip()
        logging.info(f"Loaded prompt prefixes from {file_path}.")
    except Exception as e:
        logging.error(f"Error loading prompts from {file_path}: {e}")
    return prompt_map

def ensure_dir_exists(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def detect_encoding(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read(1_000_000)
    return chardet.detect(raw)["encoding"] or "utf-8"

def load_json_list_field(json_str: str) -> List[Dict[str, Any]]:
    if not json_str:
        return []
    try:
        data = json.loads(json_str)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def build_translation_map(human_str: str, machine_str: str) -> Dict[str, str]:
    human = {
        item["language"].strip(): item["translation"].strip()
        for item in load_json_list_field(human_str)
        if item.get("language") and item.get("translation")
    }
    machine = {
        item["language"].strip(): item["translation"].strip()
        for item in load_json_list_field(machine_str)
        if item.get("language") and item.get("translation")
    }
    combined: Dict[str, str] = {}
    for lang in set(human) | set(machine):
        combined[lang] = human.get(lang) or machine.get(lang, "")
    return combined

def load_existing_results(result_file: str) -> set[tuple]:
    if not os.path.isfile(result_file):
        return set()
    enc = detect_encoding(result_file)
    done: set[tuple] = set()
    with open(result_file, "r", encoding=enc, errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((
                row.get("Updated_Entry_ID",""),
                row.get("Entry_ID",""),
                row.get("Language","")
            ))
    return done

def select_items_from_rows(rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    manual = [r for r in rows if r.get("Manual_Include","").strip().upper() == "Y"]
    non_manual = [r for r in rows if r.get("Manual_Include","").strip().upper() != "Y"]
    if len(manual) >= target:
        return manual
    remaining = target - len(manual)
    additional = random.sample(non_manual, min(remaining, len(non_manual))) if non_manual else []
    return manual + additional

###############################################################################
# ASYNC INFERENCE
###############################################################################

async def run_inference(prompt: str, engine: AsyncLLMEngine) -> str:
    sampling_params = SamplingParams(
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        ignore_eos=True,
        stop=[]
    )
    request_id = str(uuid.uuid4())
    full_text = ""
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            full_text = output.outputs[0].text
        if getattr(output, "finished", False):
            break
    return full_text.strip()

###############################################################################
# CSV PROCESSING
###############################################################################

async def process_csv_file(
    csv_file: str,
    engine: AsyncLLMEngine,
    base_corpus_dir: str,
    results_base_dir: str,
    subprompts: Dict[str, str]
):
    rel = os.path.relpath(csv_file, base_corpus_dir)
    out_csv = os.path.join(results_base_dir, rel)
    ensure_dir_exists(out_csv)

    done = load_existing_results(out_csv)
    if len(done) >= TARGET_NUM_ITEMS:
        logging.info(f"Skipping {csv_file}; {len(done)} already done.")
        return

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    selected = select_items_from_rows(rows, TARGET_NUM_ITEMS)
    logging.info(f"{len(selected)} rows selected from {csv_file}.")

    write_header = not os.path.isfile(out_csv)
    total = 0
    with open(out_csv, "a", encoding="utf-8", newline="") as fout:
        cols = [
            "Updated_Entry_ID", "Entry_ID",
            "Statement_English", "Statement_Language",
            "Time_Taken", "Reply",
            "Language", "Master_Tag", "Country"
        ]
        writer = csv.DictWriter(fout, fieldnames=cols, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()

        for row in selected:
            uid, eid = row.get("Updated_Entry_ID",""), row.get("Entry_ID","")
            eng = row.get("Entry","").strip() or row.get("Statement_English","").strip()
            country = row.get("Country","").strip("[]' ")
            master = row.get("Master_Tag","")
            trans_map = build_translation_map(
                row.get("humantranslations",""),
                row.get("machinetranslations","")
            )

            for lang, key in LANG_PROMPT_MAP.items():
                k = (uid, eid, lang)
                if k in done:
                    continue
                stmt = eng if lang == "English" else trans_map.get(lang, "")
                if not stmt:
                    logging.debug(f"No translation for {lang} in {eid}")
                    continue

                prompt = subprompts.get(key, "") + stmt
                t0 = time.time()
                reply = await run_inference(prompt, engine)
                dt = time.time() - t0

                writer.writerow({
                    "Updated_Entry_ID": uid,
                    "Entry_ID": eid,
                    "Statement_English": eng,
                    "Statement_Language": stmt,
                    "Time_Taken": f"{dt:.4f}",
                    "Reply": reply,
                    "Language": lang,
                    "Master_Tag": master,
                    "Country": country,
                })
                fout.flush()
                done.add(k)
                total += 1

    logging.info(f"Finished {csv_file}: {total} prompts.")

async def run_experiment(
    prompt_file: str,
    run_subfolder: str,
    engine: AsyncLLMEngine
):
    subprompts = load_prompts_from_file(prompt_file)
    out_dir = os.path.join(RESULTS_BASE_DIR, run_subfolder)
    logging.info(f"Experiment {prompt_file} → {out_dir}")

    for csv_file in glob.glob(os.path.join(CORPUS_DIR, "**/*.csv"), recursive=True):
        if "EMPTY" in os.path.basename(csv_file):
            continue
        await process_csv_file(csv_file, engine, CORPUS_DIR, out_dir, subprompts)

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout
    )

    engine_args = AsyncEngineArgs(
        model=MODEL_DIR,
        max_model_len=10000,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype="float16",
        pipeline_parallel_size=3,
        tensor_parallel_size=1,
        distributed_executor_backend="mp"
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    for run_num in range(1, 4):
        run_folder = f"run{run_num}"
        logging.info(f"General run: {run_folder}")
        await run_experiment(PROMPTS_FILE, run_folder, engine)

        for loc_file in glob.glob(LOCATION_PROMPTS_PATTERN):
            country = (
                os.path.basename(loc_file)
                .removeprefix("prompts_with_location_info_")
                .removesuffix(".csv")
            )
            loc_folder = f"location_run_{country}_{run_num}"
            logging.info(f"Location run: {loc_folder}")
            await run_experiment(loc_file, loc_folder, engine)

    logging.info("All experiments complete.")

if __name__ == "__main__":
    asyncio.run(main())

