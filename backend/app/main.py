from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class EventPayload(BaseModel):
    node: str
    timestamp: float = Field(default_factory=lambda: time.time())
    score: float = 0.0
    threshold: float = 0.0
    status: str = "healthy"
    kind: str = "metric"
    anomaly_type: str | None = None


@dataclass
class NodeState:
    node: str
    status: str = "offline"
    last_seen: float = 0.0
    last_score: float = 0.0
    threshold: float = 0.0
    anomaly_count: int = 0
    recent_scores: deque[float] = field(default_factory=lambda: deque(maxlen=64))


@dataclass
class DashboardStore:
    nodes: dict[str, NodeState] = field(default_factory=dict)
    alerts: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=120))

    def record_event(self, event: EventPayload) -> None:
        state = self.nodes.setdefault(event.node, NodeState(node=event.node))
        state.status = event.status
        state.last_seen = event.timestamp
        state.last_score = event.score
        state.threshold = event.threshold
        if event.kind == "alert" or event.status == "alert":
            state.anomaly_count += 1
            self.alerts.append(
                {
                    "node": event.node,
                    "timestamp": event.timestamp,
                    "score": event.score,
                    "threshold": event.threshold,
                    "anomaly_type": event.anomaly_type,
                }
            )
        state.recent_scores.append(event.score)

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        node_payload = []
        for state in self.nodes.values():
            status = state.status
            if state.last_seen and now - state.last_seen > 6.0:
                status = "offline"
            node_payload.append(
                {
                    "node": state.node,
                    "status": status,
                    "last_seen": state.last_seen,
                    "last_score": state.last_score,
                    "threshold": state.threshold,
                    "anomaly_count": state.anomaly_count,
                    "recent_scores": list(state.recent_scores),
                }
            )

        active_nodes = sum(1 for item in node_payload if item["status"] != "offline")
        alerting_nodes = sum(1 for item in node_payload if item["status"] == "alert")
        latest_alert = self.alerts[-1] if self.alerts else None
        return {
            "generated_at": now,
            "nodes": node_payload,
            "alerts": list(self.alerts),
            "summary": {
                "node_count": len(node_payload),
                "active_nodes": active_nodes,
                "alerting_nodes": alerting_nodes,
                "latest_alert": latest_alert,
            },
        }


store = DashboardStore()
app = FastAPI(title="PhantomSignal API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/events")
async def ingest_event(event: EventPayload) -> dict[str, str]:
    store.record_event(event)
    return {"status": "accepted"}


@app.get("/api/dashboard")
async def dashboard() -> dict[str, Any]:
    return store.snapshot()


@app.get("/api/nodes")
async def nodes() -> dict[str, Any]:
    snapshot = store.snapshot()
    return {"nodes": snapshot["nodes"]}
