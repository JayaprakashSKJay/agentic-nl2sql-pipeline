from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TraceSession:
    run_id: str
    out_dir: Path
    path: Path

    def log(self, event: str, data: Optional[Dict[str, Any]] = None):
        rec = {
            "ts": time.time(),
            "event": event,
            "run_id": self.run_id,
            "data": data or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def new_trace_session(base_dir: str = "./runs") -> TraceSession:
    base = Path(base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = base / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trace.jsonl"
    # create empty
    path.write_text("", encoding="utf-8")
    return TraceSession(run_id=run_id, out_dir=out_dir, path=path)
