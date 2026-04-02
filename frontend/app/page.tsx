'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  ResponsiveContainer,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type NodeSnapshot = {
  node: string;
  status: string;
  last_seen: number;
  last_score: number;
  threshold: number;
  anomaly_count: number;
  recent_scores: number[];
};

type DashboardSnapshot = {
  generated_at: number;
  nodes: NodeSnapshot[];
  alerts: Array<{
    node: string;
    timestamp: number;
    score: number;
    threshold: number;
    anomaly_type?: string | null;
  }>;
  summary: {
    node_count: number;
    active_nodes: number;
    alerting_nodes: number;
    latest_alert: {
      node: string;
      timestamp: number;
      score: number;
      threshold: number;
      anomaly_type?: string | null;
    } | null;
  };
};

type ChartPoint = {
  label: string;
  score: number;
  threshold: number;
};

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://127.0.0.1:8000';

function toChartSeries(nodes: NodeSnapshot[]): ChartPoint[] {
  const activeNode = nodes.find((node) => node.recent_scores.length > 0);
  if (!activeNode) {
    return [];
  }

  return activeNode.recent_scores.map((score, index) => ({
    label: `${activeNode.node}-${index + 1}`,
    score,
    threshold: activeNode.threshold || 0.12,
  }));
}

function statusTone(status: string): string {
  if (status === 'alert') return 'alert';
  if (status === 'offline') return 'offline';
  return 'healthy';
}

export default function Page() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('waiting for telemetry');

  useEffect(() => {
    let isMounted = true;

    async function loadSnapshot() {
      try {
        const response = await fetch(`${backendUrl}/api/dashboard`, { cache: 'no-store' });
        if (!response.ok) {
          return;
        }

        const payload = (await response.json()) as DashboardSnapshot;
        if (!isMounted) {
          return;
        }

        setSnapshot(payload);
        setLastUpdated(new Date(payload.generated_at * 1000).toLocaleTimeString());
      } catch {
        return;
      }
    }

    void loadSnapshot();
    const timer = window.setInterval(() => {
      void loadSnapshot();
    }, 1500);

    return () => {
      isMounted = false;
      window.clearInterval(timer);
    };
  }, []);

  const chartData = useMemo(() => toChartSeries(snapshot?.nodes ?? []), [snapshot]);
  const latestAlert = snapshot?.summary.latest_alert ?? null;

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">PhantomSignal / Command & Control</p>
          <h1>Distributed anomaly detection over synthetic signal streams.</h1>
          <p className="lede">
            Three edge nodes learn normal RF-like noise, flag reconstruction-error spikes, and gossip alerts peer to peer.
          </p>
        </div>

        <div className="hero-panel">
          <div>
            <span className="panel-label">Backend</span>
            <strong>{backendUrl}</strong>
          </div>
          <div>
            <span className="panel-label">Last update</span>
            <strong>{lastUpdated}</strong>
          </div>
          <div>
            <span className="panel-label">Live nodes</span>
            <strong>{snapshot?.summary.active_nodes ?? 0}</strong>
          </div>
        </div>
      </section>

      <section className="metrics-grid">
        <article className="metric-card accent">
          <span>Nodes observed</span>
          <strong>{snapshot?.summary.node_count ?? 0}</strong>
        </article>
        <article className="metric-card">
          <span>Alerting nodes</span>
          <strong>{snapshot?.summary.alerting_nodes ?? 0}</strong>
        </article>
        <article className="metric-card">
          <span>Latest anomaly score</span>
          <strong>{latestAlert ? latestAlert.score.toFixed(4) : '0.0000'}</strong>
        </article>
      </section>

      <section className="dashboard-grid">
        <article className="panel chart-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Telemetry</p>
              <h2>Reconstruction error stream</h2>
            </div>
            <div className={`alert-pill ${latestAlert ? 'flash' : ''}`}>
              {latestAlert ? `Alert on ${latestAlert.node}` : 'System steady'}
            </div>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={chartData} margin={{ top: 20, right: 24, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" strokeDasharray="4 4" />
                <XAxis dataKey="label" tick={{ fill: 'rgba(226,232,240,0.72)', fontSize: 12 }} interval="preserveStartEnd" />
                <YAxis tick={{ fill: 'rgba(226,232,240,0.72)', fontSize: 12 }} width={40} />
                <Tooltip
                  contentStyle={{
                    background: '#07111f',
                    border: '1px solid rgba(255,255,255,0.12)',
                    borderRadius: 16,
                    color: '#e2e8f0',
                  }}
                />
                <Line type="monotone" dataKey="score" stroke="#8ff2c6" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="threshold" stroke="#fb7185" strokeWidth={2} strokeDasharray="6 6" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel nodes-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Swarm</p>
              <h2>Node health</h2>
            </div>
          </div>

          <div className="node-list">
            {(snapshot?.nodes ?? []).map((node) => (
              <div className={`node-card ${statusTone(node.status)}`} key={node.node}>
                <div className="node-top">
                  <strong>{node.node}</strong>
                  <span>{node.status}</span>
                </div>
                <div className="node-body">
                  <div>
                    <span>Last score</span>
                    <strong>{node.last_score.toFixed(4)}</strong>
                  </div>
                  <div>
                    <span>Alerts</span>
                    <strong>{node.anomaly_count}</strong>
                  </div>
                  <div>
                    <span>Threshold</span>
                    <strong>{node.threshold.toFixed(4)}</strong>
                  </div>
                </div>
              </div>
            ))}
            {(!snapshot || snapshot.nodes.length === 0) && <p className="empty-state">Waiting for nodes to report telemetry.</p>}
          </div>
        </article>
      </section>

      <section className="panel alerts-panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">C2 Feed</p>
            <h2>Recent alerts</h2>
          </div>
        </div>

        <div className="alert-list">
          {(snapshot?.alerts ?? []).slice().reverse().map((alert) => (
            <div className="alert-row" key={`${alert.node}-${alert.timestamp}`}>
              <div>
                <strong>{alert.node}</strong>
                <p>{alert.anomaly_type ?? 'reconstruction spike'}</p>
              </div>
              <div className="alert-score">{alert.score.toFixed(4)}</div>
            </div>
          ))}
          {(!snapshot || snapshot.alerts.length === 0) && <p className="empty-state">No alerts yet.</p>}
        </div>
      </section>
    </main>
  );
}
