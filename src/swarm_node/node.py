from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from src.edge_ai.autoencoder import AutoencoderDetector


@dataclass(slots=True)
class GossipMessage:
    message_id: str
    source: str
    kind: str
    timestamp: float
    payload: dict[str, Any]

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "message_id": self.message_id,
                "source": self.source,
                "kind": self.kind,
                "timestamp": self.timestamp,
                "payload": self.payload,
            }
        ).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "GossipMessage":
        payload = json.loads(raw.decode("utf-8"))
        return cls(
            message_id=payload["message_id"],
            source=payload["source"],
            kind=payload["kind"],
            timestamp=float(payload["timestamp"]),
            payload=dict(payload["payload"]),
        )


class GossipTransport(asyncio.DatagramProtocol):
    def __init__(self, node_name: str, peers: list[tuple[str, int]], on_message: callable[[GossipMessage], Any]) -> None:
        self.node_name = node_name
        self.peers = peers
        self.on_message = on_message
        self.transport: asyncio.DatagramTransport | None = None
        self.seen_ids: set[str] = set()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            message = GossipMessage.from_json(data)
        except Exception:
            return

        if message.message_id in self.seen_ids:
            return
        self.seen_ids.add(message.message_id)
        self.on_message(message)
        asyncio.create_task(self.broadcast(message, exclude=addr))

    async def broadcast(self, message: GossipMessage, exclude: tuple[str, int] | None = None) -> None:
        if self.transport is None:
            return
        packet = message.to_json()
        for peer in self.peers:
            if exclude is not None and peer == exclude:
                continue
            self.transport.sendto(packet, peer)

    async def send_alert(self, payload: dict[str, Any]) -> None:
        message = GossipMessage(
            message_id=str(uuid.uuid4()),
            source=self.node_name,
            kind="alert",
            timestamp=time.time(),
            payload=payload,
        )
        self.seen_ids.add(message.message_id)
        await self.broadcast(message)


class SwarmNode:
    def __init__(
        self,
        node_name: str,
        stream_host: str,
        stream_port: int,
        backend_url: str,
        gossip_host: str,
        gossip_port: int,
        peers: list[tuple[str, int]],
        train_frames: int,
        threshold_scale: float,
        min_threshold: float,
        max_frames: int | None,
    ) -> None:
        self.node_name = node_name
        self.stream_host = stream_host
        self.stream_port = stream_port
        self.backend_url = backend_url.rstrip("/")
        self.gossip_host = gossip_host
        self.gossip_port = gossip_port
        self.peers = peers
        self.train_frames = train_frames
        self.threshold_scale = threshold_scale
        self.min_threshold = min_threshold
        self.max_frames = max_frames

        self.detector: AutoencoderDetector | None = None
        self.gossip: GossipTransport | None = None
        self.http = httpx.AsyncClient(timeout=3.0)
        self.errors: deque[float] = deque(maxlen=64)

    async def run(self) -> None:
        print(f"[{self.node_name}] connecting stream {self.stream_host}:{self.stream_port}")
        reader, writer = await asyncio.open_connection(self.stream_host, self.stream_port)
        try:
            await self._start_gossip()
            await self._send_event(kind="status", status="training", score=0.0, threshold=0.0, anomaly_type=None)
            trained_on = await self._train_from_stream(reader)
            print(f"[{self.node_name}] trained on {trained_on} normal frames, threshold={self.detector.threshold:.5f}")
            await self._send_event(kind="status", status="healthy", score=0.0, threshold=self.detector.threshold, anomaly_type=None)
            await self._detection_loop(reader)
        finally:
            writer.close()
            await writer.wait_closed()
            await self.http.aclose()

    async def _start_gossip(self) -> None:
        loop = asyncio.get_running_loop()
        self.gossip = GossipTransport(self.node_name, self.peers, self._on_gossip)
        await loop.create_datagram_endpoint(lambda: self.gossip, local_addr=(self.gossip_host, self.gossip_port))
        print(f"[{self.node_name}] gossip listening on {self.gossip_host}:{self.gossip_port}")

    async def _train_from_stream(self, reader: asyncio.StreamReader) -> int:
        windows: list[np.ndarray] = []
        sample_size: int | None = None
        while len(windows) < self.train_frames:
            raw = await reader.readline()
            if not raw:
                break
            frame = json.loads(raw.decode("utf-8"))
            samples = np.asarray(frame["samples"], dtype=np.float32)
            if sample_size is None:
                sample_size = int(samples.shape[0])
            if frame.get("inject_anomaly"):
                continue
            windows.append(samples)

        if not windows:
            raise RuntimeError("Could not collect normal training frames from stream")
        sample_size = sample_size or int(windows[0].shape[0])
        self.detector = AutoencoderDetector(
            window_size=sample_size,
            threshold_scale=self.threshold_scale,
            min_threshold=self.min_threshold,
        )
        training_windows = np.stack(windows)
        self.detector.fit(training_windows)
        return len(windows)

    async def _detection_loop(self, reader: asyncio.StreamReader) -> None:
        assert self.detector is not None
        count = 0
        while True:
            raw = await reader.readline()
            if not raw:
                print(f"[{self.node_name}] stream ended")
                return
            frame = json.loads(raw.decode("utf-8"))
            samples = np.asarray(frame["samples"], dtype=np.float32)
            score, alert = self.detector.score(samples)
            self.errors.append(score)

            status = "alert" if alert else "healthy"
            anomaly_type = frame.get("anomaly_type")
            await self._send_event(
                kind="alert" if alert else "metric",
                status=status,
                score=score,
                threshold=self.detector.threshold,
                anomaly_type=anomaly_type,
            )

            if alert and self.gossip is not None:
                await self.gossip.send_alert(
                    {
                        "summary": f"{self.node_name} reconstruction error spike",
                        "score": score,
                        "threshold": self.detector.threshold,
                        "anomaly_type": anomaly_type,
                    }
                )
                print(
                    f"[{self.node_name}] ALERT seq={frame['seq']} score={score:.5f} threshold={self.detector.threshold:.5f} labeled={frame.get('inject_anomaly')}"
                )
            else:
                print(
                    f"[{self.node_name}] metric seq={frame['seq']} score={score:.5f} threshold={self.detector.threshold:.5f} labeled={frame.get('inject_anomaly')}"
                )

            count += 1
            if self.max_frames is not None and count >= self.max_frames:
                return

    def _on_gossip(self, message: GossipMessage) -> None:
        if message.kind != "alert":
            return
        print(f"[{self.node_name}] gossip from {message.source}: {message.payload.get('summary', 'alert')}")

    async def _send_event(self, kind: str, status: str, score: float, threshold: float, anomaly_type: str | None) -> None:
        payload = {
            "node": self.node_name,
            "timestamp": time.time(),
            "score": score,
            "threshold": threshold,
            "status": status,
            "kind": kind,
            "anomaly_type": anomaly_type,
        }
        try:
            await self.http.post(f"{self.backend_url}/api/events", json=payload)
        except Exception:
            return


def parse_peers(raw_peers: list[str]) -> list[tuple[str, int]]:
    peers: list[tuple[str, int]] = []
    for item in raw_peers:
        host, port = item.split(":", maxsplit=1)
        peers.append((host, int(port)))
    return peers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PhantomSignal swarm node over live stream frames.")
    parser.add_argument("--node", required=True, help="Node name, for example Alpha")
    parser.add_argument("--stream-host", default="127.0.0.1")
    parser.add_argument("--stream-port", type=int, default=8765)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gossip-host", default="127.0.0.1")
    parser.add_argument("--gossip-port", type=int, required=True)
    parser.add_argument("--peer", action="append", default=[], help="Peer as host:port. Repeat for multiple peers.")
    parser.add_argument("--train-frames", type=int, default=120)
    parser.add_argument("--threshold-scale", type=float, default=3.0)
    parser.add_argument("--min-threshold", type=float, default=0.01)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    node = SwarmNode(
        node_name=args.node,
        stream_host=args.stream_host,
        stream_port=args.stream_port,
        backend_url=args.backend_url,
        gossip_host=args.gossip_host,
        gossip_port=args.gossip_port,
        peers=parse_peers(args.peer),
        train_frames=args.train_frames,
        threshold_scale=args.threshold_scale,
        min_threshold=args.min_threshold,
        max_frames=args.max_frames,
    )
    try:
        asyncio.run(node.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
