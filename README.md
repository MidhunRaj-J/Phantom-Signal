# PhantomSignal

PhantomSignal is a software-only MVP for a distributed signal-intelligence simulator. It models the shape of an edge-defense system without requiring hardware radios or a live RF environment. The goal is to prove the full workflow in software first: generate believable signal data, detect abnormal patterns with a small neural model, gossip alerts between nodes, and visualize the result in a command-and-control dashboard.

## Why This Exists

This project is intentionally not “just another AI demo.” The point is to show a systems-oriented prototype that combines signal processing, anomaly detection, peer-to-peer coordination, and a live operator UI.

The value of the MVP is in the architecture, not in a perfect model score. It demonstrates that:

- Synthetic sensor data can stand in for expensive or unavailable hardware during early development.
- A lightweight unsupervised model can detect deviations from normal signal behavior.
- Edge nodes can coordinate without a central coordinator by broadcasting alerts directly to peers.
- A single dashboard can surface telemetry, alert history, and node health in near real time.

## What Exists Today

The repository currently contains a working end-to-end software MVP:

- A Python backend built with FastAPI for ingesting node telemetry and serving dashboard snapshots.
- A synthetic signal generator that produces normal RF-like windows and injects anomalies.
- A 1D convolutional autoencoder that learns the normal signal profile and scores reconstruction error.
- A UDP-based gossip layer that lets edge nodes broadcast alert messages to peers.
- A Next.js frontend with live charts, node status cards, and a recent-alert feed.

In other words, the repo already contains the core loop from signal creation through detection and visualization.

## What Is New In This MVP

The current build adds the pieces needed to turn the concept into a coherent demo:

- Synthetic RF-like stream generation using Gaussian noise, sinusoidal carriers, burst injections, and frequency-hopping style anomalies.
- An unsupervised 1D convolutional autoencoder trained only on normal windows.
- Reconstruction-error based anomaly scoring with a threshold learned from normal samples.
- A small edge-node runtime that can be launched in multiple terminals to simulate Alpha, Beta, and Charlie as separate devices.
- Peer-to-peer alert sharing over UDP so an alert on one node can be observed by the others without a central message broker.
- A dark, operator-style dashboard that presents score trends, node health, and recent alerts in one view.

## How The System Fits Together

1. The backend generates synthetic windows that look like noisy RF data.
2. Each node trains or loads the detector and scores incoming windows.
3. If the reconstruction error crosses the threshold, the node emits an alert.
4. The alert is sent both to the backend for dashboard display and to peer nodes over gossip.
5. The frontend polls the backend and shows the current swarm state in the browser.

## Repository Layout

- `backend/` - FastAPI app, signal generator, detector, gossip transport, and node runtime.
- `frontend/` - Next.js dashboard with live charts and node status.
- `.github/copilot-instructions.md` - workspace instructions used for future Copilot work.

## Backend

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Run the API:

```powershell
uvicorn app.main:app --reload --app-dir backend
```

Run three edge nodes in separate terminals:

```powershell
cd backend
python -m app.node --node Alpha --listen-port 9101 --peer 127.0.0.1:9102 --peer 127.0.0.1:9103
python -m app.node --node Beta --listen-port 9102 --peer 127.0.0.1:9101 --peer 127.0.0.1:9103
python -m app.node --node Charlie --listen-port 9103 --peer 127.0.0.1:9101 --peer 127.0.0.1:9102
```

The backend exposes:

- `GET /health` - basic health check.
- `POST /api/events` - ingest node telemetry and alerts.
- `GET /api/dashboard` - snapshot used by the frontend.
- `GET /api/nodes` - node-only snapshot for simpler integrations.

## Frontend

Install dependencies and start the dashboard:

```powershell
cd frontend
npm install
npm run dev
```

If the API is not running at `http://127.0.0.1:8000`, set `NEXT_PUBLIC_BACKEND_URL` before starting the frontend.

## Current Behaviors

- Generates normal signal windows from Gaussian noise plus a few stable carrier tones.
- Injects two anomaly shapes: high-energy bursts and frequency-hopping style transitions.
- Scores each window with a compact autoencoder and compares the result against a learned threshold.
- Broadcasts alert payloads between nodes with a lightweight JSON gossip message.
- Keeps a rolling history of scores and alerts for dashboard visualization.
- Presents node health as healthy, alert, or offline depending on recent telemetry.

## Development Notes

- The frontend uses handcrafted CSS instead of a utility framework, so the UI stays intentionally opinionated without a large styling dependency chain.
- The backend is intentionally simple and local-first. It simulates distributed behavior on a single machine so the architecture can be demonstrated before any hardware or cloud work.
- `npm run build` currently passes in `frontend/`, and the backend syntax checks cleanly with `py -3 -m compileall backend` on Windows.

## Suggested Next Steps

1. Replace UDP gossip with a more durable transport if you want persistence or multi-host testing.
2. Add a persisted event store so the dashboard can replay sessions after the nodes stop.
3. Separate model training from inference so the detector can be reused without retraining on startup.
4. Add automated tests for the signal generator, detector thresholding, and dashboard snapshot API.
