# -*- coding: utf-8 -*-
"""
Thread-safe in-memory LRU and TTL trace storage for RAG diagnostics.
Ensures bounded retention without external database dependencies.
"""

import threading
import time
import os
from collections import OrderedDict
from typing import Dict, Optional, List, Any
from .schema import TraceRecord


class TraceStore:
    """
    In-memory thread-safe store for diagnostic traces with max capacity (LRU) and TTL.
    """
    def __init__(self, max_capacity: int = 500, ttl_seconds: float = 86400.0):
        if max_capacity <= 0:
            raise ValueError("max_capacity must be positive")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds cannot be negative")
        self.max_capacity = int(max_capacity)
        self.ttl_seconds = float(ttl_seconds)
        self._store: OrderedDict[str, TraceRecord] = OrderedDict()
        self._lock = threading.RLock()

    def put(self, trace: TraceRecord) -> None:
        """Store a trace record, evicting oldest if max capacity is exceeded."""
        with self._lock:
            self._cleanup_expired()
            if trace.trace_id in self._store:
                self._store.move_to_end(trace.trace_id)
            self._store[trace.trace_id] = trace
            if len(self._store) > self.max_capacity:
                self._store.popitem(last=False)

    def get(self, trace_id: str) -> Optional[TraceRecord]:
        """Retrieve a trace record by trace_id."""
        with self._lock:
            self._cleanup_expired()
            trace = self._store.get(trace_id)
            if trace is not None:
                self._store.move_to_end(trace_id)
            return trace

    def list_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List summary of stored traces."""
        with self._lock:
            self._cleanup_expired()
            results = []
            for tid, t in reversed(self._store.items()):
                results.append({
                    "trace_id": t.trace_id,
                    "timestamp": t.timestamp,
                    "query_type": t.query_type,
                    "query": t.query.original_query if t.query else "",
                    "answer_preview": t.generation.answer[:100] if t.generation else "",
                    "error_count": len(t.errors),
                    "total_latency_ms": t.timing.total_latency_ms if t.timing else 0.0,
                })
                if len(results) >= limit:
                    break
            return results

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            self._cleanup_expired()
            return len(self._store)

    def _cleanup_expired(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        expired_keys = [
            k for k, v in self._store.items() if now - v.timestamp > self.ttl_seconds
        ]
        for k in expired_keys:
            self._store.pop(k, None)


# Global singleton trace store.  The limits are intentionally process-local and
# configurable so a long-running web worker cannot grow without bound.
def _positive_env_number(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


_GLOBAL_TRACE_STORE = TraceStore(
    max_capacity=_positive_env_int("RAG_DIAGNOSTIC_MAX_TRACES", 500),
    ttl_seconds=_positive_env_number("RAG_DIAGNOSTIC_TTL_SECONDS", 86400.0),
)


def get_global_trace_store() -> TraceStore:
    return _GLOBAL_TRACE_STORE
