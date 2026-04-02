from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .generator import generate_rf_stream


class ChunkRecorder:
    """Writes stream windows to chunked artifacts for later model training."""

    def __init__(self, output_dir: Path, chunk_size: int = 256, mode: str = "npz") -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.mode = mode
        self._signals: list[np.ndarray] = []
        self._labels: list[int] = []
        self._meta: list[dict[str, Any]] = []
        self._chunk_index = 0

    def append(self, frame: dict[str, Any]) -> None:
        self._signals.append(np.asarray(frame["samples"], dtype=np.float32))
        self._labels.append(1 if frame["inject_anomaly"] else 0)
        self._meta.append(
            {
                "seq": frame["seq"],
                "timestamp": frame["timestamp"],
                "anomaly_type": frame["anomaly_type"],
                "rms": frame["stats"]["rms"],
                "peak": frame["stats"]["peak"],
            }
        )
        if len(self._signals) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._signals:
            return
        signals = np.stack(self._signals)
        labels = np.asarray(self._labels, dtype=np.int8)
        meta = np.asarray(self._meta, dtype=object)

        if self.mode == "parquet":
            self._write_parquet(signals=signals, labels=labels, meta=meta)
        else:
            self._write_npz(signals=signals, labels=labels, meta=meta)

        self._signals.clear()
        self._labels.clear()
        self._meta.clear()
        self._chunk_index += 1

    def _write_npz(self, signals: np.ndarray, labels: np.ndarray, meta: np.ndarray) -> None:
        target = self.output_dir / f"rf_chunk_{self._chunk_index:05d}.npz"
        np.savez_compressed(target, signals=signals, labels=labels, meta=meta)

    def _write_parquet(self, signals: np.ndarray, labels: np.ndarray, meta: np.ndarray) -> None:
        try:
            import pandas as pd
        except Exception:
            # Fallback to NPZ if optional parquet stack is missing.
            self._write_npz(signals=signals, labels=labels, meta=meta)
            return

        target = self.output_dir / f"rf_chunk_{self._chunk_index:05d}.parquet"
        frame = pd.DataFrame(meta.tolist())
        frame["label"] = labels
        frame["samples"] = [sample.tolist() for sample in signals]
        frame.to_parquet(target, index=False)


class SignalStreamServer:
    def __init__(
        self,
        host: str,
        port: int,
        frame_hz: float,
        samples_per_frame: int,
        anomaly_rate: float,
        record_to: Path | None,
        record_mode: str,
        record_chunk_size: int,
        seed: int | None,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_hz = frame_hz
        self.samples_per_frame = samples_per_frame
        self.anomaly_rate = anomaly_rate
        self.clients: set[asyncio.StreamWriter] = set()
        self.seq = 0
        self.rng = np.random.default_rng(seed)
        self.recorder = (
            ChunkRecorder(output_dir=record_to, chunk_size=record_chunk_size, mode=record_mode)
            if record_to is not None
            else None
        )

    async def start(self) -> None:
        server = await asyncio.start_server(self._on_client, self.host, self.port)
        sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
        print(f"[signal-engine] listening on {sockets}")
        async with server:
            producer = asyncio.create_task(self._producer_loop())
            try:
                await server.serve_forever()
            finally:
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer

    async def _on_client(self, _: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.clients.add(writer)
        peer = writer.get_extra_info("peername")
        print(f"[signal-engine] client connected: {peer}")
        try:
            while not writer.is_closing():
                await asyncio.sleep(0.5)
        finally:
            self.clients.discard(writer)
            writer.close()
            await writer.wait_closed()
            print(f"[signal-engine] client disconnected: {peer}")

    async def _producer_loop(self) -> None:
        delay = 1.0 / max(self.frame_hz, 0.1)
        while True:
            frame = self._build_frame()
            encoded = (json.dumps(frame) + "\n").encode("utf-8")
            stale_writers: list[asyncio.StreamWriter] = []
            for writer in self.clients:
                try:
                    writer.write(encoded)
                    await writer.drain()
                except Exception:
                    stale_writers.append(writer)
            for stale in stale_writers:
                self.clients.discard(stale)

            if self.recorder is not None:
                self.recorder.append(frame)

            if frame["inject_anomaly"]:
                print(
                    f"[signal-engine] anomaly frame seq={frame['seq']} rms={frame['stats']['rms']:.4f} type={frame['anomaly_type']}"
                )

            await asyncio.sleep(delay)

    def _build_frame(self) -> dict[str, Any]:
        inject_anomaly = float(self.rng.random()) < self.anomaly_rate
        time_axis, signal = generate_rf_stream(samples=self.samples_per_frame, inject_anomaly=inject_anomaly)
        anomaly_type = "burst" if inject_anomaly else None
        frame = {
            "seq": self.seq,
            "timestamp": time.time(),
            "inject_anomaly": inject_anomaly,
            "anomaly_type": anomaly_type,
            "time_axis": time_axis.tolist(),
            "samples": signal.tolist(),
            "stats": {
                "rms": float(np.sqrt(np.mean(np.square(signal)))),
                "peak": float(np.max(np.abs(signal))),
            },
        }
        self.seq += 1
        return frame

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 synthetic RF stream server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--frame-hz", type=float, default=8.0, help="Frames emitted per second")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per generated frame")
    parser.add_argument("--anomaly-rate", type=float, default=0.16, help="Probability that a frame includes threat injection")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument("--record-dir", type=Path, default=None, help="Optional capture directory for generated frames")
    parser.add_argument("--record-mode", choices=["npz", "parquet"], default="npz")
    parser.add_argument("--record-chunk-size", type=int, default=256)
    return parser


async def run_server(args: argparse.Namespace) -> None:
    server = SignalStreamServer(
        host=args.host,
        port=args.port,
        frame_hz=args.frame_hz,
        samples_per_frame=args.samples,
        anomaly_rate=args.anomaly_rate,
        record_to=args.record_dir,
        record_mode=args.record_mode,
        record_chunk_size=args.record_chunk_size,
        seed=args.seed,
    )
    try:
        await server.start()
    finally:
        server.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
