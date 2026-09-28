"""
Assignment 11 — Audit log for request and response decisions.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start an audit entry and return the id to pair with its output."""
        request_id = request_id or uuid.uuid4().hex
        self._open[request_id] = {
            "request_id": request_id,
            "user_id": user_id or "anonymous",
            "input": text or "",
            "input_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete an audit entry with the decision and elapsed time."""
        pending_id = request_id
        if pending_id is None:
            pending_id = next(
                (
                    key
                    for key, entry in reversed(list(self._open.items()))
                    if entry["user_id"] == (user_id or "anonymous")
                ),
                None,
            )

        pending = self._open.pop(pending_id, None) if pending_id else None
        now = utc_now_iso()
        elapsed_ms = (
            (time.perf_counter() - pending["started_monotonic"]) * 1000
            if pending
            else 0.0
        )
        entry = {
            "request_id": pending_id or uuid.uuid4().hex,
            "user_id": (pending or {}).get("user_id", user_id or "anonymous"),
            "input": (pending or {}).get("input", ""),
            "input_at": (pending or {}).get("input_at", now),
            "output": text or "",
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": round(elapsed_ms, 3),
            "output_at": now,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
