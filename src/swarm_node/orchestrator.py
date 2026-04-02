from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class ProcSpec:
    name: str
    args: list[str]


async def spawn_process(spec: ProcSpec) -> asyncio.subprocess.Process:
    print(f"[orchestrator] starting {spec.name}")
    return await asyncio.create_subprocess_exec(*spec.args)


async def wait_and_teardown(processes: list[tuple[ProcSpec, asyncio.subprocess.Process]]) -> None:
    stop = asyncio.Event()

    def _request_stop(*_: object) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # add_signal_handler can be unsupported on some event loops.
            pass

    await stop.wait()
    print("[orchestrator] stopping child processes")
    for _, proc in processes:
        if proc.returncode is None:
            proc.terminate()

    await asyncio.gather(*(proc.wait() for _, proc in processes), return_exceptions=True)


def build_specs(args: argparse.Namespace) -> list[ProcSpec]:
    py = sys.executable
    stream_spec = ProcSpec(
        name="signal-engine",
        args=[
            py,
            "-m",
            "src.signal_engine.stream_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.stream_port),
            "--frame-hz",
            str(args.frame_hz),
            "--anomaly-rate",
            str(args.anomaly_rate),
        ],
    )

    node_specs = [
        ProcSpec(
            name="node-alpha",
            args=[
                py,
                "-m",
                "src.swarm_node.node",
                "--node",
                "Alpha",
                "--stream-port",
                str(args.stream_port),
                "--backend-url",
                args.backend_url,
                "--gossip-port",
                "9201",
                "--peer",
                "127.0.0.1:9202",
                "--peer",
                "127.0.0.1:9203",
                "--train-frames",
                str(args.train_frames),
            ],
        ),
        ProcSpec(
            name="node-beta",
            args=[
                py,
                "-m",
                "src.swarm_node.node",
                "--node",
                "Beta",
                "--stream-port",
                str(args.stream_port),
                "--backend-url",
                args.backend_url,
                "--gossip-port",
                "9202",
                "--peer",
                "127.0.0.1:9201",
                "--peer",
                "127.0.0.1:9203",
                "--train-frames",
                str(args.train_frames),
            ],
        ),
        ProcSpec(
            name="node-charlie",
            args=[
                py,
                "-m",
                "src.swarm_node.node",
                "--node",
                "Charlie",
                "--stream-port",
                str(args.stream_port),
                "--backend-url",
                args.backend_url,
                "--gossip-port",
                "9203",
                "--peer",
                "127.0.0.1:9201",
                "--peer",
                "127.0.0.1:9202",
                "--train-frames",
                str(args.train_frames),
            ],
        ),
    ]

    return [stream_spec, *node_specs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch stream server plus three swarm nodes for local demo.")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--stream-port", type=int, default=8765)
    parser.add_argument("--frame-hz", type=float, default=8.0)
    parser.add_argument("--anomaly-rate", type=float, default=0.16)
    parser.add_argument("--train-frames", type=int, default=120)
    return parser


async def main_async(args: argparse.Namespace) -> None:
    specs = build_specs(args)
    launched: list[tuple[ProcSpec, asyncio.subprocess.Process]] = []
    for spec in specs:
        proc = await spawn_process(spec)
        launched.append((spec, proc))
        await asyncio.sleep(0.4)

    print("[orchestrator] all processes launched. Press Ctrl+C to stop.")
    await wait_and_teardown(launched)


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
