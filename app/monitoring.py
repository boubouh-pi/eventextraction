"""
Request monitoring: append-only JSONL log + aggregation for /metrics.

JSONL over SQLite: records are immutable facts written once and read
rarely (only by /metrics), a single writer process appends under a lock,
and the file is trivially greppable/shippable to any log stack. SQLite
would add locking semantics and a schema migration story for no gain at
this scale.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RequestRecord:
    timestamp: float
    site: Optional[str]
    n_nodes: int
    n_candidates: int
    n_events: int
    extraction_method: str
    llm_model: Optional[str]
    llm_cost_usd: float
    latency_ms: float
    status: str = "ok"
    error: Optional[str] = None


class RequestLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record: RequestRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # tolerate a torn write, don't kill /metrics
        return records

    def aggregate(self) -> dict:
        records = self._read_all()
        if not records:
            return {
                "total_requests": 0,
                "total_events_extracted": 0,
                "total_llm_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "avg_nodes_per_request": 0.0,
                "method_breakdown": {},
                "requests_by_site": {},
            }

        methods: dict[str, int] = {}
        sites: dict[str, int] = {}
        for r in records:
            methods[r.get("extraction_method", "?")] = methods.get(r.get("extraction_method", "?"), 0) + 1
            site = r.get("site") or "unknown"
            sites[site] = sites.get(site, 0) + 1

        n = len(records)
        return {
            "total_requests": n,
            "total_events_extracted": sum(r.get("n_events", 0) for r in records),
            "total_llm_cost_usd": round(sum(r.get("llm_cost_usd", 0.0) for r in records), 6),
            "avg_latency_ms": round(sum(r.get("latency_ms", 0.0) for r in records) / n, 2),
            "avg_nodes_per_request": round(sum(r.get("n_nodes", 0) for r in records) / n, 1),
            "method_breakdown": methods,
            "requests_by_site": sites,
        }


def now_ms() -> float:
    return time.perf_counter() * 1000.0
