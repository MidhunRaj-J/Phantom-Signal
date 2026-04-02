from __future__ import annotations

import argparse
import asyncio
import json
from collections import deque
from typing import Any

import numpy as np


class EdgeStreamConsumer:
    """Consumes live signal frames and raises simple anomaly warnings.

    This is a Phase 1.5 stub before wiring in the PyTorch autoencoder.
    """

    def __init__(self, host: str, port: int, max_frames: int | None = None) -> None:
        self.host = host
        self.port = port
        self.max_frames = max_frames
        self.processed = 0
        self.rms_history: deque[float] = deque(maxlen=80)

    async def run(self) -> None:
        print(f"[edge-ai] connecting to {self.host}:{self.port}")
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                frame = json.loads(raw.decode("utf-8"))
                self._process_frame(frame)
                self.processed += 1
                if self.max_frames is not None and self.processed >= self.max_frames:
                    break
        finally:
            writer.close()
            await writer.wait_closed()

    def _process_frame(self, frame: dict[str, Any]) -> None:
        samples = np.asarray(frame["samples"], dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples))))
        self.rms_history.append(rms)
        baseline = float(np.mean(self.rms_history)) if self.rms_history else rms
        spike_threshold = baseline * 1.45
        is_spike = rms > spike_threshold and len(self.rms_history) > 12

        if is_spike:
            print(
                f"[edge-ai] ALERT seq={frame['seq']} rms={rms:.4f} baseline={baseline:.4f} labeled={frame['inject_anomaly']}"
            )
        else:
            print(
                f"[edge-ai] frame seq={frame['seq']} rms={rms:.4f} baseline={baseline:.4f} labeled={frame['inject_anomaly']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consume synthetic RF stream as an edge AI stub.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame count before stopping")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(EdgeStreamConsumer(args.host, args.port, args.max_frames).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
