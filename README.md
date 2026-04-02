# PhantomSignal

PhantomSignal is a software-only MVP for decentralized signal intelligence. It simulates RF-like telemetry, performs edge anomaly detection with an unsupervised 1D autoencoder, shares alerts through peer-to-peer gossip, and presents live operational status in a command-and-control dashboard.

## Executive Summary

PhantomSignal demonstrates an end-to-end system architecture rather than an isolated model demo.

- Signal-level simulation: realistic noisy streams plus controlled threat injection.
- Edge AI: local unsupervised inference using reconstruction error.
- Decentralized coordination: node-to-node gossip without a central alert broker.
- Operator visibility: FastAPI telemetry API + Next.js real-time dashboard.

This makes the project relevant to modern edge defense, distributed sensing, and resilient systems engineering.

## Why This Project Matters

Most popular stacks solve only one layer:

- AI assistants and RAG systems optimize language workflows, not low-level telemetry behavior.
- Centralized SOC/SIEM stacks assume normalized data already exists in one place.
- Many anomaly detection projects are notebook-first and batch-oriented.
- Device platforms often prioritize fleet management over autonomous peer collaboration.

PhantomSignal closes this gap by combining generation, detection, coordination, and visualization in one coherent system.

## Architecture At A Glance

1. Signal engine produces synthetic RF-like windows.
2. Edge nodes train on normal windows and infer on live frames.
3. Reconstruction error crossing a threshold triggers an alert.
4. Alerts are gossiped to peers and posted to the backend API.
5. Dashboard renders node health, scores, and alert history in real time.

## Technology Stack

- Python: core simulation, detector, swarm runtime, and orchestration.
- PyTorch: 1D convolutional autoencoder.
- NumPy / SciPy: synthetic signal construction and numeric processing.
- FastAPI + Uvicorn: telemetry ingestion and dashboard snapshot API.
- Next.js + React + Recharts: live C2 dashboard.
- Async IO + UDP gossip: decentralized peer alert propagation.

## Repository Structure

- `backend/`: FastAPI backend, API store, legacy node runtime.
- `frontend/`: Next.js dashboard UI.
- `src/signal_engine/`: generator and live stream server.
- `src/edge_ai/`: autoencoder and stream consumer tooling.
- `src/swarm_node/`: swarm node runtime and orchestrator.
- `src/c2_dashboard/`: reserved for future dashboard adapters.
- `data/`: optional captured stream chunks for training.
- `docs/`: architecture diagrams and project artifacts.

## Current Implemented Capabilities

- Synthetic Gaussian-noise stream with carrier components.
- Threat injection (burst anomalies; extensible anomaly patterns).
- TCP newline-delimited JSON stream transport.
- Optional chunked capture (`npz`, optional parquet fallback behavior).
- Autoencoder warmup training from normal live frames.
- Reconstruction-error thresholding and alert emission.
- UDP gossip propagation across swarm peers.
- Backend event ingestion and node/alert aggregation.
- Real-time dashboard visualization of node state and anomalies.
- Single-command orchestrator for stream + Alpha/Beta/Charlie nodes.

## Prerequisites

- Windows, macOS, or Linux with Python 3.11+.
- Node.js 20+ and npm.

Install Python dependencies from repository root:

```powershell
py -3 -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
Set-Location frontend
npm install
```

## Full-System Quickstart (Recommended)

Run each step in a separate terminal.

1. Start backend API:

```powershell
Set-Location c:\Projects\RAG
py -3 -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

2. Start dashboard:

```powershell
Set-Location c:\Projects\RAG\frontend
npm run dev
```

3. Start stream + swarm nodes:

```powershell
Set-Location c:\Projects\RAG
py -3 -m src.swarm_node.orchestrator --backend-url http://127.0.0.1:8000 --stream-port 8765 --frame-hz 8 --anomaly-rate 0.16 --train-frames 120
```

4. Open:

- `http://localhost:3000`

Expected result: live Alpha, Beta, and Charlie telemetry with score updates and alert transitions.

## Component-Level Commands

Run generator visualization:

```powershell
Set-Location c:\Projects\RAG
py -3 src\signal_engine\generator.py
```

Run standalone stream server:

```powershell
Set-Location c:\Projects\RAG
py -3 -m src.signal_engine.stream_server --host 127.0.0.1 --port 8765 --frame-hz 8 --anomaly-rate 0.16
```

Run stream server with capture:

```powershell
Set-Location c:\Projects\RAG
py -3 -m src.signal_engine.stream_server --record-dir data --record-mode npz --record-chunk-size 256
```

Run one swarm node manually:

```powershell
Set-Location c:\Projects\RAG
py -3 -m src.swarm_node.node --node Alpha --stream-port 8765 --backend-url http://127.0.0.1:8000 --gossip-port 9201 --peer 127.0.0.1:9202 --peer 127.0.0.1:9203 --train-frames 120
```

## API Endpoints

- `GET /health`: backend health check.
- `POST /api/events`: ingest metric and alert events.
- `GET /api/dashboard`: aggregated dashboard snapshot.
- `GET /api/nodes`: node-only status snapshot.

## Troubleshooting

- Port `8000` already in use:
  Start backend once, or change backend port and update orchestrator/frontend target URL.
- `npm run dev` fails from repository root:
  Run it from `frontend/`.
- Port `3000` already in use:
  Next.js may move to `3001`; open the URL printed by the dev server.

## Validation Status

- `py -3 -m compileall src` passes.
- `py -3 -m src.swarm_node.node --help` passes.
- `py -3 -m src.swarm_node.orchestrator --help` passes.
- Frontend production build passes in `frontend/`.

## Roadmap

1. Persist trained weights to skip warmup on every restart.
2. Add durable event storage and dashboard replay.
3. Harden gossip with authentication and encryption.
4. Add automated tests for stream framing, model thresholds, gossip propagation, and API snapshots.
