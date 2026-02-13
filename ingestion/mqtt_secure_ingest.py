from __future__ import annotations

import argparse
import json
import ssl
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from paho.mqtt import client as mqtt_client

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_COLUMN_NAMES = [
    "unit_nr",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"s_{i}" for i in range(1, 22)]
DEFAULT_WINDOW_LENGTH = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure MQTT subscriber for real-time RUL inference.")
    parser.add_argument("--broker", required=True, help="MQTT broker host.")
    parser.add_argument("--port", type=int, default=8883, help="MQTT TLS port.")
    parser.add_argument("--topic", required=True, help="Subscribed topic.")
    parser.add_argument("--client-id", default="rul-secure-ingest")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--ca-cert", type=Path, required=True)
    parser.add_argument("--client-cert", type=Path)
    parser.add_argument("--client-key", type=Path)
    parser.add_argument("--model-mode", default="Baseline")
    parser.add_argument("--min-cycles", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument("--event-log", type=Path, default=Path("logs/mqtt_events.ndjson"))
    parser.add_argument("--prediction-log", type=Path, default=Path("logs/mqtt_predictions.csv"))
    return parser.parse_args()


def _write_event(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")


def _append_prediction(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if not path.exists():
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)


def _normalize_row(row: Dict[str, object], required_columns: List[str]) -> Dict[str, float | int]:
    normalized: Dict[str, float | int] = {}
    for col in required_columns:
        if col not in row:
            raise ValueError(f"Incoming message missing required field: {col}")
        normalized[col] = row[col]  # type: ignore[assignment]
    return normalized


def run() -> None:
    args = parse_args()
    from backend.model_service import ModelService

    service = ModelService()
    by_engine: Dict[int, List[Dict[str, float | int]]] = defaultdict(list)

    client = mqtt_client.Client(client_id=args.client_id, protocol=mqtt_client.MQTTv311)
    if args.username:
        client.username_pw_set(args.username, args.password)

    tls_kwargs = {
        "ca_certs": str(args.ca_cert),
        "cert_reqs": ssl.CERT_REQUIRED,
        "tls_version": ssl.PROTOCOL_TLS_CLIENT,
    }
    if args.client_cert and args.client_key:
        tls_kwargs["certfile"] = str(args.client_cert)
        tls_kwargs["keyfile"] = str(args.client_key)
    client.tls_set(**tls_kwargs)

    def on_connect(client_obj, _userdata, _flags, rc):
        if rc != 0:
            print(f"[mqtt] connect failed rc={rc}")
            return
        print(f"[mqtt] connected to {args.broker}:{args.port}, subscribing to {args.topic}")
        client_obj.subscribe(args.topic, qos=1)

    def on_message(_client_obj, _userdata, msg):
        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            row = _normalize_row(payload, RAW_COLUMN_NAMES)
            engine_id = int(row["unit_nr"])
            by_engine[engine_id].append(row)
            _write_event(
                args.event_log,
                {
                    "timestamp_utc": now,
                    "topic": msg.topic,
                    "qos": int(msg.qos),
                    "engine_id": engine_id,
                    "payload": row,
                },
            )
            if len(by_engine[engine_id]) < max(args.min_cycles, DEFAULT_WINDOW_LENGTH):
                return

            engine_df = pd.DataFrame(by_engine[engine_id], columns=RAW_COLUMN_NAMES)
            csv_bytes = engine_df.to_csv(index=False).encode("utf-8")
            result = service.infer(csv_bytes, model_mode=args.model_mode)
            rel = result["per_engine_reliability"][engine_id]
            pred = float(result["per_engine_mean_rul"][engine_id])
            out = {
                "timestamp_utc": now,
                "engine_id": engine_id,
                "model_mode": args.model_mode,
                "predicted_rul": pred,
                "trusted_rul": float(rel["trusted_rul"]),
                "ri": float(rel["ri"]),
                "decision": rel["decision"],
            }
            _append_prediction(args.prediction_log, out)
            print(json.dumps(out))
        except Exception as exc:  # pragma: no cover - runtime ingestion guard.
            print(f"[mqtt] message handling error: {exc}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    run()
