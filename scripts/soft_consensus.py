#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import glob
import json
import re
import logging
import fcntl
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

###############################################################################
# SIMPLE FILE LOCK
###############################################################################
class FileLock:
    def __init__(self, lockfile):
        self.lockfile = lockfile
        self.fd = None

    def __enter__(self):
        self.fd = open(self.lockfile, "a+")
        fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
RESULTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "consensus.log")

MAX_WORKERS = min(256, os.cpu_count() or 1)
VALID_STATUS = {"moderated", "unmoderated"}

# regex to extract exactly the moderation_status field
MOD_STATUS_RE = re.compile(
    r'"moderation_status"\s*:\s*"(moderated|unmoderated)"'
)

# setup logger at ERROR level only
logger = logging.getLogger("consensus")
logger.setLevel(logging.ERROR)
fh = RotatingFileHandler(LOG_PATH, maxBytes=10_000_000, backupCount=3)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(fh)

###############################################################################
# WORKER: PROCESS A SINGLE CSV
###############################################################################
def process_file(path: str):

    lockpath = path + ".lock"
    with FileLock(lockpath):
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return

            orig_fields = list(rows[0].keys())
            had_consensus = "consensus_soft" in orig_fields
            had_blocked = "blocked_response" in orig_fields

            fieldnames = orig_fields[:]
            if not had_consensus: fieldnames.append("consensus_soft")
            if not had_blocked: fieldnames.append("blocked_response")

            updated = 0

            for idx, row in enumerate(rows, start=1):
                if row.get("blocked_response","").strip().lower() == "true":
                    continue
                if row.get("consensus_soft","").strip():
                    continue

                eng = row.get("EnglishTranslation","")
                api_raw = row.get("API_Response","").strip()
                if not eng and api_raw:
                    try:
                        api_obj = json.loads(api_raw)
                        prompt_feedback = api_obj.get("prompt_feedback")
                        block_reason = prompt_feedback.get("block_reason") if prompt_feedback else None
                    except json.JSONDecodeError:
                        raise ValueError(f"{path}:{idx} invalid API_Response JSON")
                    if block_reason:
                        for c in [
                            "openai_label","mistral_label","gemini_label",
                            "consensus_label","deberta_soft_label",
                            "openai_soft_label","gemini_soft_label"
                        ]:
                            if c in row:
                                row[c] = ""
                        row["blocked_response"] = "true"
                        updated += 1
                        continue

                votes = []
                d = row.get("deberta_soft_label","").strip().lower()
                if d in VALID_STATUS:
                    votes.append(d)

                def parse_soft(cell: str, prefix: str):
                    if not cell.strip():
                        return None
                    try:
                        val = json.loads(cell).get("moderation_status")
                    except json.JSONDecodeError:
                        m = MOD_STATUS_RE.search(cell)
                        val = m.group(1) if m else None
                    if val in VALID_STATUS:
                        return val
                    return None

                o_stat = parse_soft(row.get("openai_soft_label",""), "OpenAI")
                if o_stat:
                    votes.append(o_stat)

                g_stat = parse_soft(row.get("gemini_soft_label",""), "Gemini")
                if g_stat:
                    votes.append(g_stat)

                if len(votes) == 0:
                    continue
                elif len(votes) == 1:
                    consensus = votes[0]
                else:
                    mod = votes.count("moderated")
                    unmod = votes.count("unmoderated")
                    if mod > unmod:
                        consensus = "moderated"
                    elif unmod > mod:
                        consensus = "unmoderated"
                    else:
                        consensus = votes[0]

                row["consensus_soft"] = consensus
                updated += 1

            need_header = not (had_consensus and had_blocked)
            if updated > 0 or need_header:
                with open(path, "r+", newline="", encoding="utf-8") as f:
                    f.seek(0)
                    writer = csv.DictWriter(f, fieldnames=fieldnames,
                                            quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                    writer.writerows(rows)
                    f.truncate()

        except Exception as e:
            logger.error(f"{path}: fatal error: {e}")
            raise

###############################################################################
# MAIN
###############################################################################
def main():
    pattern = os.path.join(RESULTS_BASE_DIR, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    
    if len(files) == 1:
        try:
            process_file(files[0])
            print(f"Successfully processed: {files[0]}")
        except Exception as e:
            print(f"Error processing {files[0]}: {e}")
            sys.exit(1)
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = {exe.submit(process_file, p): p for p in files}
            for fut in tqdm(as_completed(futures), total=len(files), desc="Files", unit="file"):
                try:
                    fut.result()
                except:
                    sys.exit(1)

if __name__ == "__main__":
    main()

