"""
Interaction logging (VBS_GUIDE.md section 7) - explicitly called out as a
"first-class requirement" of the competition, not an afterthought: the
annual joint post-competition analysis paper depends on every team's
submitted logs, and in the VBS 2023 analysis several teams' logs were
unusable (unrecoverable timestamps, incomplete records) and those teams
were DROPPED from the analysis entirely.

Every entry is written to a local JSONL file FIRST - this always succeeds
unless disk is unwritable, so nothing is lost even if DRES's log endpoint
is down, slow, or misconfigured - then best-effort pushed to DRES via
dres_client.submit_query_log/submit_interaction_log. A failed DRES push
is silently swallowed (those functions already return None rather than
raising) and never blocks or fails the caller's actual response; the
local JSONL file remains the source of truth that can be re-submitted
later if needed.
"""
import json
import time
from pathlib import Path
from typing import Optional

import dres_client

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "interaction_log.jsonl"


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _append_local(entry: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_query(
    action: str,
    query: str,
    result_ids: list,
    dres_config: Optional[dict] = None,
    session_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Records one query-issuing action - `action` names which kind
    ("search" / "feedback" / "query_by_example" / "temporal_search"),
    `query` is the text query (or a short description for non-text
    actions like query-by-example), `result_ids` is the top-k point ids
    actually shown to the operator (VBS_GUIDE.md section 7 requires at
    least the top-k retrieved items per query in the log).
    """
    entry = {
        "type": "query",
        "action": action,
        "query": query,
        "result_ids": result_ids,
        "timestamp_ms": _timestamp_ms(),
    }
    if extra:
        entry.update(extra)
    _append_local(entry)

    if dres_config and dres_config.get("base_url") and session_id and dres_config.get("evaluation_id"):
        dres_client.submit_query_log(dres_config["base_url"], session_id, dres_config["evaluation_id"], entry)
    return entry


def log_interaction(
    action: str,
    details: dict,
    dres_config: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Records a user action that isn't itself a new query - e.g. submitting
    an answer to DRES, browsing a video, playing a clip. Same
    local-first, best-effort-DRES-push contract as log_query.
    """
    entry = {
        "type": "interaction",
        "action": action,
        "details": details,
        "timestamp_ms": _timestamp_ms(),
    }
    _append_local(entry)

    if dres_config and dres_config.get("base_url") and session_id and dres_config.get("evaluation_id"):
        dres_client.submit_interaction_log(dres_config["base_url"], session_id, dres_config["evaluation_id"], entry)
    return entry
