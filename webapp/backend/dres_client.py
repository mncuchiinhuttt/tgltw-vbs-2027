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
from typing import Optional


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
    payload: dict,
    timeout: float = 10.0,
) -> dict:
    """
    Submit an answer for the current task. `payload` shape depends on task
    type (VBS_GUIDE.md section 4) and must be built by the caller:
      - KIS (KIS-V/T/C): a video item + a temporal range or single frame
        (e.g. {"mediaItemName": ..., "start": ms, "end": ms}).
      - AVS: a list of media items (one submission per shot found).
      - VQA: free-text answer (e.g. {"text": "..."}).
    These shapes are inferred from the guide's description of DRES's
    submission model, not yet verified against a live OpenAPI spec - build
    a small helper per task type once the real schema is confirmed rather
    than constructing payload dicts ad hoc at call sites.
    """
    url = f"{base_url.rstrip('/')}/api/v2/submit/{evaluation_id}"
    try:
        resp = requests.post(
            url, params={"session": session_id}, json={"taskId": task_id, **payload}, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise DresError(f"DRES submit_answer failed: {e}") from e
    except ValueError as e:
        raise DresError(f"DRES submit_answer returned a non-JSON response: {e}") from e


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
