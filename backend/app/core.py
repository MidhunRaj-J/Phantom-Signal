from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import numpy as np
import torch
from scipy import signal
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class SyntheticSignalGenerator:
    def __init__(self, window_size: int = 256, sample_rate: int = 256) -> None:
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.time_axis = np.linspace(0.0, 1.0, window_size, endpoint=False)

    def normal_window(self, rng: np.random.Generator) -> np.ndarray:
        carriers = np.zeros(self.window_size, dtype=np.float32)
        frequencies = (5.0, 17.0, 31.0)
        amplitudes = (0.65, 0.35, 0.2)

        for frequency, amplitude in zip(frequencies, amplitudes, strict=True):
            phase = rng.uniform(0.0, 2.0 * math.pi)
            carriers += amplitude * np.sin(2.0 * math.pi * frequency * self.time_axis + phase)

        low_drift = 0.12 * np.sin(2.0 * math.pi * 0.7 * self.time_axis + rng.uniform(0.0, 2.0 * math.pi))
        noise = rng.normal(0.0, 0.28, self.window_size)
        return (carriers + low_drift + noise).astype(np.float32)

    def inject_anomaly(self, window: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        anomalous = window.copy()
        kind = rng.choice(["burst", "frequency_hop"])

        if kind == "burst":
            burst_center = rng.integers(self.window_size // 5, self.window_size - self.window_size // 5)
            burst_width = rng.integers(10, 24)
            burst_slice = slice(max(0, burst_center - burst_width), min(self.window_size, burst_center + burst_width))
            burst_time = np.linspace(-1.0, 1.0, burst_slice.stop - burst_slice.start)
            envelope = np.exp(-6.0 * burst_time**2)
            carrier = np.sin(2.0 * math.pi * rng.uniform(18.0, 42.0) * burst_time)
            anomalous[burst_slice] += (2.6 * envelope * carrier).astype(np.float32)
        else:
            hop_point = rng.integers(self.window_size // 4, self.window_size - self.window_size // 4)
            pre = self.time_axis[:hop_point]
            post = self.time_axis[hop_point:]
            hop_a = signal.chirp(pre, f0=rng.uniform(4.0, 8.0), t1=1.0, f1=rng.uniform(12.0, 18.0), method="linear")
            hop_b = signal.chirp(post, f0=rng.uniform(22.0, 28.0), t1=1.0, f1=rng.uniform(42.0, 58.0), method="quadratic")
            anomalous[:hop_point] += (0.9 * hop_a).astype(np.float32)
            anomalous[hop_point:] += (1.7 * hop_b).astype(np.float32)

        return anomalous.astype(np.float32), kind


class SignalDataset(Dataset):
    def __init__(self, windows: np.ndarray) -> None:
        self.data = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


class SignalAutoencoder(nn.Module):
    def __init__(self, window_size: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1),
        )
        self.window_size = window_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        if decoded.shape[-1] != self.window_size:
            decoded = torch.nn.functional.interpolate(decoded, size=self.window_size, mode="linear", align_corners=False)
        return decoded


class SignalDetector:
    def __init__(self, window_size: int = 256, device: str | None = None) -> None:
        self.window_size = window_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SignalAutoencoder(window_size=window_size).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.threshold = 0.12

    def fit(self, generator: SyntheticSignalGenerator, epochs: int = 4, samples: int = 768, batch_size: int = 32) -> float:
        rng = np.random.default_rng(42)
        windows = np.stack([generator.normal_window(rng) for _ in range(samples)])
        dataset = SignalDataset(windows)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstructed = self.model(batch)
                loss = self.loss_fn(reconstructed, batch)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            self.model.eval()
            sampled_errors = []
            for window in windows[: min(len(windows), 128)]:
                sampled_errors.append(self.reconstruction_error(window))
            self.threshold = float(np.mean(sampled_errors) + 3.0 * np.std(sampled_errors))
            self.threshold = max(self.threshold, 0.06)
        return self.threshold

    def reconstruction_error(self, window: np.ndarray) -> float:
        tensor = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(tensor)
            error = torch.mean((reconstruction - tensor) ** 2)
        return float(error.item())

    def is_anomalous(self, window: np.ndarray) -> tuple[float, bool]:
        score = self.reconstruction_error(window)
        return score, score >= self.threshold


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
        data = json.loads(raw.decode("utf-8"))
        return cls(
            message_id=data["message_id"],
            source=data["source"],
            kind=data["kind"],
            timestamp=float(data["timestamp"]),
            payload=dict(data["payload"]),
        )


class BackendReporter:
    def __init__(self, backend_url: str) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=3.0)

    async def send_event(self, event: dict[str, Any]) -> None:
        try:
            await self.client.post(f"{self.backend_url}/api/events", json=event)
        except Exception:
            return

    async def close(self) -> None:
        await self.client.aclose()


class UdpGossipTransport(asyncio.DatagramProtocol):
    def __init__(self, node_name: str, peer_addresses: list[tuple[str, int]], on_message: Callable[[GossipMessage], Any]) -> None:
        super().__init__()
        self.node_name = node_name
        self.peer_addresses = peer_addresses
        self.on_message = on_message
        self.transport: asyncio.DatagramTransport | None = None
        self.seen_messages: set[str] = set()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            message = GossipMessage.from_json(data)
        except Exception:
            return

        if message.message_id in self.seen_messages:
            return
        self.seen_messages.add(message.message_id)
        self.on_message(message)
        asyncio.create_task(self.broadcast(message, exclude=addr))

    async def broadcast(self, message: GossipMessage, exclude: tuple[str, int] | None = None) -> None:
        if self.transport is None:
            return
        payload = message.to_json()
        for peer in self.peer_addresses:
            if exclude is not None and peer == exclude:
                continue
            self.transport.sendto(payload, peer)

    async def send_alert(self, payload: dict[str, Any]) -> None:
        message = GossipMessage(
            message_id=str(uuid.uuid4()),
            source=self.node_name,
            kind="alert",
            timestamp=time.time(),
            payload=payload,
        )
        self.seen_messages.add(message.message_id)
        await self.broadcast(message)


class PhantomSignalNode:
    def __init__(
        self,
        node_name: str,
        listen_host: str,
        listen_port: int,
        peer_addresses: list[tuple[str, int]],
        backend_url: str,
        alert_threshold: float = 0.12,
        anomaly_rate: float = 0.18,
    ) -> None:
        self.node_name = node_name
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.peer_addresses = peer_addresses
        self.backend = BackendReporter(backend_url)
        self.generator = SyntheticSignalGenerator()
        self.detector = SignalDetector(window_size=self.generator.window_size)
        self.alert_threshold = alert_threshold
        self.anomaly_rate = anomaly_rate
        self.rng = np.random.default_rng(abs(hash(node_name)) % (2**32))
        self.recent_scores: deque[float] = deque(maxlen=48)
        self.transport: UdpGossipTransport | None = None
        self.status = "booting"

    async def start(self) -> None:
        self.status = "training"
        threshold = self.detector.fit(self.generator)
        self.alert_threshold = max(self.alert_threshold, threshold)
        loop = asyncio.get_running_loop()
        self.transport = UdpGossipTransport(self.node_name, self.peer_addresses, self._on_gossip_message)
        await loop.create_datagram_endpoint(lambda: self.transport, local_addr=(self.listen_host, self.listen_port))
        self.status = "healthy"

    def _on_gossip_message(self, message: GossipMessage) -> None:
        if message.kind != "alert":
            return
        print(
            f"[{self.node_name}] gossip received from {message.source}: {message.payload.get('summary', 'alert')}"
        )

    async def run(self) -> None:
        await self.start()
        await self.backend.send_event(self._status_event())

        while True:
            await self.step()
            await asyncio.sleep(0.35)

    async def step(self) -> dict[str, Any]:
        signal_window = self.generator.normal_window(self.rng)
        anomaly_label = None
        if float(self.rng.random()) < self.anomaly_rate:
            signal_window, anomaly_label = self.generator.inject_anomaly(signal_window, self.rng)

        score, is_anomaly = self.detector.is_anomalous(signal_window)
        self.recent_scores.append(score)

        event = {
            "node": self.node_name,
            "timestamp": time.time(),
            "score": score,
            "threshold": self.alert_threshold,
            "status": "alert" if is_anomaly else "healthy",
            "kind": "alert" if is_anomaly else "metric",
            "anomaly_type": anomaly_label,
        }
        await self.backend.send_event(event)

        if is_anomaly and self.transport is not None:
            print(f"[{self.node_name}] anomaly detected: score={score:.4f}, type={anomaly_label}")
            await self.transport.send_alert(
                {
                    "summary": f"High reconstruction error on {self.node_name}",
                    "score": score,
                    "threshold": self.alert_threshold,
                    "anomaly_type": anomaly_label,
                }
            )
        else:
            print(f"[{self.node_name}] score={score:.4f}")

        return event

    def _status_event(self) -> dict[str, Any]:
        return {
            "node": self.node_name,
            "timestamp": time.time(),
            "score": 0.0,
            "threshold": self.alert_threshold,
            "status": self.status,
            "kind": "status",
        }


def parse_peer_addresses(raw_peers: list[str]) -> list[tuple[str, int]]:
    peers: list[tuple[str, int]] = []
    for peer in raw_peers:
        host, port = peer.split(":", maxsplit=1)
        peers.append((host, int(port)))
    return peers
