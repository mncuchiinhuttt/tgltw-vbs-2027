"""
Thin wrapper around DRES (Distributed Retrieval Evaluation Server), the
evaluation infrastructure VBS runs on (VBS_GUIDE.md section 6). DRES's own
recommended workflow is to generate a typed client from its OpenAPI spec
(https://raw.githubusercontent.com/dres-dev/DRES/master/doc/oas-client.json,
pinned to whatever release the competition runs) rather than hand-writing
REST calls - the guide's worked example does this for Angular/TypeScript.

We don't have a live DRES instance to develop against in this sandbox, and
by VBS 2027 the deployed version will likely be newer than the guide's
pinned 2.0.1 example - so this module hand-rolls the calls against DRES's
documented endpoint shapes as a starting point. TREAT THIS AS UNVERIFIED:
once the competition's actual DRES version/instance is known, either
regenerate a typed Python client (e.g. via `openapi-python-client generate
--url <oas-client.json for that version>`) and swap it in behind the same
functions below, or hit a real instance and fix up whatever paths/payload
shapes turn out to differ.

Every function here is a plain, stateless helper - no module-level session
state - callers (webapp/backend/main.py) hold the session token and pass it
into each call, matching how a single-operator-per-instance webapp uses it.
"""
import requests
from typing import List, Optional


class DresError(Exception):
    """Raised when a DRES call fails - callers should surface this as a
    clear error rather than silently treating the competition server as
    unreachable."""


def login(base_url: str, username: str, password: str, timeout: float = 10.0) -> str:
    """
    POST /api/v2/login -> session id.
    DRES's v2 API returns a session identifier in the JSON body (commonly
    "sessionId"); older/newer versions may use a cookie instead - if this
    endpoint 404s against the real instance, check whether the competition
    is running the v1 API path (`/api/login`) instead.
    """
    url = f"{base_url.rstrip('/')}/api/v2/login"
    try:
        resp = requests.post(url, json={"username": username, "password": password}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise DresError(f"DRES login failed: {e}") from e
    except ValueError as e:
        # Non-JSON body (HTML proxy page, empty reply) - JSONDecodeError is a
        # RequestException subclass on requests >= 2.27, ValueError below.
        raise DresError(f"DRES login returned a non-JSON response: {e}") from e
    session_id = data.get("sessionId") or data.get("session_id")
    if not session_id:
        raise DresError(f"DRES login response missing sessionId: {data}")
    return session_id


def get_current_task(base_url: str, session_id: str, evaluation_id: str, timeout: float = 10.0) -> dict:
    """
    GET the current task for a running evaluation. Returns a dict with at
    least {task_id, type, hints, time_remaining} once verified against a
    real instance - field names are a best guess from the guide's
    description (section 6/13), not yet confirmed against a live response.
    """
    url = f"{base_url.rstrip('/')}/api/v2/client/evaluation/currentTask/{evaluation_id}"
    try:
        resp = requests.get(url, params={"session": session_id}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise DresError(f"DRES get_current_task failed: {e}") from e
    except ValueError as e:
        raise DresError(f"DRES get_current_task returned a non-JSON response: {e}") from e


def submit_answer(
    base_url: str,
    session_id: str,
    evaluation_id: str,
    task_id: str,
    answers: List[dict],
    timeout: float = 10.0,
) -> dict:
    """
    Submit answers for the current task - DRES 2.x submission contract:

        body = {"taskId": ..., "answers": [AnswerWrite, ...]}

    Each AnswerWrite is either a media-item range in MILLISECONDS
    ({"mediaItemName": ..., "start": ms, "end": ms}) for KIS/AVS/KIS-V, or
    a plain text answer ({"text": ...}) for VQA/QA-style tasks, or a list
    of media-item ranges (one per segment, in order) for TRAKE-style
    multi-segment tasks. This replaces the earlier flat
    {"mediaItemName", "timestamp"-in-seconds} guess that no DRES version
    accepts. The shape still needs a rehearsal against the live instance -
    call sites build AnswerWrite entries via the router's
    _build_dres_answers, which owns the seconds->milliseconds conversion.
    """
    url = f"{base_url.rstrip('/')}/api/v2/submit/{evaluation_id}"
    try:
        resp = requests.post(
            url, params={"session": session_id}, json={"taskId": task_id, "answers": answers}, timeout=timeout
        )
        if resp.status_code >= 400:
            # DRES's validation detail (which field was wrong, why) lives in
            # the response body - surfacing it is how an operator recovers
            # inside a 5-minute task instead of blind-retrying.
            raise DresError(
                f"DRES submit rejected ({resp.status_code}): {resp.text[:500]}"
            )
        return resp.json()
    except requests.RequestException as e:
        raise DresError(f"DRES submit_answer failed: {e}") from e
    except ValueError as e:
        raise DresError(f"DRES submit_answer returned a non-JSON response: {e}") from e


def parse_submission_verdict(data: dict) -> str:
    """
    Best-effort extraction of a per-answer verdict from a DRES SubmitResponse
    so the console can show CORRECT/WRONG instead of raw JSON. DRES reports
    each answer's evaluation as a status/verdict field; the exact casing has
    not been verified against a live instance, so this scans defensively and
    falls back to a raw-JSON preview.
    """
    answers = data.get("answers") if isinstance(data, dict) else None
    if isinstance(answers, list) and answers:
        verdicts = []
        for a in answers:
            v = None
            if isinstance(a, dict):
                v = a.get("status") or a.get("verdict") or a.get("result")
            verdicts.append(str(v) if v is not None else "UNKNOWN")
        return ", ".join(verdicts)
    return str(data)[:200]


def submit_query_log(
    base_url: str,
    session_id: str,
    evaluation_id: str,
    query_log: dict,
    timeout: float = 10.0,
) -> Optional[dict]:
    """
    Submit a query log entry (VBS_GUIDE.md section 7 - query specifications,
    timestamps, team/user ids). Best-effort: callers should NOT let a
    failure here block the actual search response - log locally first
    (see interaction_log.py, added in a later phase) and treat this as a
    background push that may fail without losing the record entirely.
    """
    url = f"{base_url.rstrip('/')}/api/v2/log/query/{evaluation_id}"
    try:
        resp = requests.post(url, params={"session": session_id}, json=query_log, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def submit_interaction_log(
    base_url: str,
    session_id: str,
    evaluation_id: str,
    interaction_log: dict,
    timeout: float = 10.0,
) -> Optional[dict]:
    """
    Submit an interaction log entry (result lists shown, user actions -
    VBS_GUIDE.md section 7). Same best-effort contract as
    submit_query_log: returns None on failure instead of raising, since a
    lost log push should never break the operator's live search flow.
    """
    url = f"{base_url.rstrip('/')}/api/v2/log/result/{evaluation_id}"
    try:
        resp = requests.post(url, params={"session": session_id}, json=interaction_log, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None
