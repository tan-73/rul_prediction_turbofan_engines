import { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const COMPARE_MODE = "Compare (Baseline vs PI)";

function parseCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) {
    throw new Error("CSV has no data rows.");
  }
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      const raw = (values[i] ?? "").trim();
      const asNum = Number(raw);
      row[h] = Number.isFinite(asNum) && raw !== "" ? asNum : raw;
    });
    return row;
  });
  return rows;
}

function decisionBadge(decision) {
  if (decision === "ACCEPT") return "tag tag-accept";
  if (decision === "WARN") return "tag tag-warn";
  if (decision === "REJECT") return "tag tag-reject";
  return "tag";
}

export default function App() {
  const [mode, setMode] = useState("Baseline");
  const [rows, setRows] = useState([]);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [inferResult, setInferResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [replayRows, setReplayRows] = useState([]);
  const [replayEngine, setReplayEngine] = useState(1);
  const [replayStep, setReplayStep] = useState(1);

  const availableEngines = useMemo(() => {
    if (inferResult?.engine_ids?.length) return inferResult.engine_ids;
    if (compareResult?.baseline?.engine_ids?.length) return compareResult.baseline.engine_ids;
    return [];
  }, [inferResult, compareResult]);

  async function onFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseCsv(text);
      setRows(parsed);
      setFileName(file.name);
      setError("");
      setInferResult(null);
      setCompareResult(null);
      setReplayRows([]);
    } catch (err) {
      setError(String(err));
    }
  }

  async function runInference() {
    if (!rows.length) {
      setError("Upload a CSV before running inference.");
      return;
    }
    setLoading(true);
    setError("");
    setReplayRows([]);
    try {
      if (mode === COMPARE_MODE) {
        const resp = await fetch(`${API_BASE}/v1/compare/json`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rows }),
        });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.detail || "Compare request failed.");
        setCompareResult(payload);
        setInferResult(null);
      } else {
        const resp = await fetch(`${API_BASE}/v1/infer/json`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rows, model_mode: mode }),
        });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.detail || "Inference request failed.");
        setInferResult(payload);
        setCompareResult(null);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function runReplay() {
    if (!rows.length) {
      setError("Upload a CSV before running replay.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const effectiveMode = mode === COMPARE_MODE ? "Baseline" : mode;
      const resp = await fetch(`${API_BASE}/v1/replay/json`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rows,
          model_mode: effectiveMode,
          engine_id: Number(replayEngine),
          step: Number(replayStep),
        }),
      });
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.detail || "Replay request failed.");
      setReplayRows(payload.rows || []);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>RUL Ops Dashboard</h1>
        <p>FastAPI + React interface for Baseline/PI inference, compare, and replay.</p>
      </header>

      <section className="panel controls">
        <div className="control-grid">
          <label>
            CSV Input
            <input type="file" accept=".csv,.txt" onChange={onFileChange} />
          </label>
          <label>
            Model Mode
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option>Baseline</option>
              <option>Physics-Informed</option>
              <option>{COMPARE_MODE}</option>
            </select>
          </label>
          <div className="buttons">
            <button onClick={runInference} disabled={loading}>
              {loading ? "Running..." : "Run Inference"}
            </button>
          </div>
        </div>
        <small>Loaded file: {fileName || "none"}</small>
        {error && <div className="error">{error}</div>}
      </section>

      {inferResult && (
        <section className="panel">
          <h2>Inference Summary</h2>
          <div className="cards">
            <div className="card">
              <span>Overall RUL</span>
              <strong>{Number(inferResult.overall_mean_rul || 0).toFixed(2)}</strong>
            </div>
            <div className="card">
              <span>Overall RI</span>
              <strong>{Number(inferResult.overall_reliability_index || 0).toFixed(3)}</strong>
            </div>
            <div className="card">
              <span>Engines</span>
              <strong>{inferResult.engine_ids?.length || 0}</strong>
            </div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Engine</th>
                <th>Pred RUL</th>
                <th>RI</th>
                <th>Decision</th>
                <th>Trusted RUL</th>
              </tr>
            </thead>
            <tbody>
              {(inferResult.engine_ids || []).map((engineId) => {
                const rel = inferResult.per_engine_reliability?.[engineId] || {};
                return (
                  <tr key={engineId}>
                    <td>{engineId}</td>
                    <td>{Number(inferResult.per_engine_mean_rul?.[engineId] || 0).toFixed(2)}</td>
                    <td>{Number(rel.ri || 0).toFixed(3)}</td>
                    <td>
                      <span className={decisionBadge(rel.decision)}>{rel.decision}</span>
                    </td>
                    <td>{Number(rel.trusted_rul || 0).toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      {compareResult && (
        <section className="panel">
          <h2>Baseline vs PI Compare</h2>
          <div className="cards">
            <div className="card">
              <span>Baseline RUL</span>
              <strong>{Number(compareResult.baseline?.overall_mean_rul || 0).toFixed(2)}</strong>
            </div>
            <div className="card">
              <span>PI RUL</span>
              <strong>{Number(compareResult.physics_informed?.overall_mean_rul || 0).toFixed(2)}</strong>
            </div>
            <div className="card">
              <span>Engine Deltas</span>
              <strong>{compareResult.engine_deltas?.length || 0}</strong>
            </div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Engine</th>
                <th>Baseline RUL</th>
                <th>PI RUL</th>
                <th>RUL Delta</th>
                <th>RI Delta</th>
                <th>Decision Delta</th>
              </tr>
            </thead>
            <tbody>
              {(compareResult.engine_deltas || []).map((row) => (
                <tr key={row.engine_id}>
                  <td>{row.engine_id}</td>
                  <td>{Number(row.baseline_pred_rul || 0).toFixed(2)}</td>
                  <td>{Number(row.pi_pred_rul || 0).toFixed(2)}</td>
                  <td>{Number(row.pred_rul_delta_pi_minus_baseline || 0).toFixed(2)}</td>
                  <td>{Number(row.ri_delta_pi_minus_baseline || 0).toFixed(3)}</td>
                  <td>{row.decision_delta}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="panel">
        <h2>Streaming Replay</h2>
        <div className="control-grid replay-grid">
          <label>
            Engine
            <select
              value={replayEngine}
              onChange={(e) => setReplayEngine(Number(e.target.value))}
              disabled={!availableEngines.length}
            >
              {(availableEngines.length ? availableEngines : [1]).map((eid) => (
                <option key={eid}>{eid}</option>
              ))}
            </select>
          </label>
          <label>
            Step
            <input
              type="number"
              min="1"
              max="10"
              value={replayStep}
              onChange={(e) => setReplayStep(Number(e.target.value))}
            />
          </label>
          <div className="buttons">
            <button onClick={runReplay} disabled={loading}>
              Run Replay
            </button>
          </div>
        </div>
        {!!replayRows.length && (
          <table>
            <thead>
              <tr>
                <th>Cycle</th>
                <th>Pred RUL</th>
                <th>Trusted RUL</th>
                <th>RI</th>
                <th>Decision</th>
              </tr>
            </thead>
            <tbody>
              {replayRows.map((r, idx) => (
                <tr key={`${r.time_cycles}-${idx}`}>
                  <td>{r.time_cycles}</td>
                  <td>{Number(r.predicted_rul || 0).toFixed(2)}</td>
                  <td>{Number(r.trusted_rul || 0).toFixed(2)}</td>
                  <td>{Number(r.reliability_index || 0).toFixed(3)}</td>
                  <td>
                    <span className={decisionBadge(r.decision)}>{r.decision}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

