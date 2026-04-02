from __future__ import annotations

import argparse
import asyncio

from .core import PhantomSignalNode, parse_peer_addresses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PhantomSignal edge node.")
    parser.add_argument("--node", required=True, help="Node name, for example Alpha")
    parser.add_argument("--listen-host", default="127.0.0.1", help="Host to bind for gossip")
    parser.add_argument("--listen-port", type=int, required=True, help="UDP port for gossip")
    parser.add_argument(
        "--peer",
        action="append",
        default=[],
        help="Peer address in host:port form. Repeat for multiple peers.",
    )
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000", help="FastAPI backend URL")
    parser.add_argument("--threshold", type=float, default=0.12, help="Alert threshold override")
    parser.add_argument("--anomaly-rate", type=float, default=0.18, help="Probability of synthetic anomaly")
    return parser


async def main_async() -> None:
    parser = build_parser()
    args = parser.parse_args()
    node = PhantomSignalNode(
        node_name=args.node,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        peer_addresses=parse_peer_addresses(args.peer),
        backend_url=args.backend_url,
        alert_threshold=args.threshold,
        anomaly_rate=args.anomaly_rate,
    )
    try:
        await node.run()
    finally:
        await node.backend.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
