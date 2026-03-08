from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from paho.mqtt import client as mqtt_client

RAW_COLUMN_NAMES = [
    "unit_nr",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"s_{i}" for i in range(1, 22)]

DEGRADING_SENSORS = ["s_4", "s_7", "s_11", "s_12", "s_15"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Digital twin MQTT publisher that mimics turbofan sensor streams."
    )
    parser.add_argument("--seed-csv", type=Path, default=Path("examples/sample_cmapss_engine.csv"))
    parser.add_argument(
        "--digital-twin-root",
        type=Path,
        default=Path("digital-twin/digital-twin-for-aircraft-engine-maintenance"),
        help="Optional reference repo path (used for contextual linkage only).",
    )
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="engines/fd001/raw")
    parser.add_argument("--client-id", default="rul-digital-twin")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--engine-id", type=int, default=1)
    parser.add_argument("--interval-sec", type=float, default=0.5)
    parser.add_argument("--cycles", type=int, default=300)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--degradation-rate", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qos", type=int, default=0, choices=[0, 1, 2])
    return parser.parse_args()


def _load_template(seed_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(seed_csv)
    missing = [c for c in RAW_COLUMN_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Seed CSV missing columns: {missing}")
    return df[RAW_COLUMN_NAMES].copy().reset_index(drop=True)


def _next_row(
    template_row: dict,
    cycle: int,
    engine_id: int,
    noise_scale: float,
    degradation_rate: float,
) -> dict:
    row = dict(template_row)
    row["unit_nr"] = int(engine_id)
    row["time_cycles"] = int(cycle)

    for sensor in [f"s_{i}" for i in range(1, 22)]:
        base = float(row[sensor])
        noisy = base + random.gauss(0.0, abs(base) * noise_scale + 1e-4)
        if sensor in DEGRADING_SENSORS:
            noisy = noisy + cycle * degradation_rate * (1.0 + random.uniform(-0.1, 0.1))
        row[sensor] = float(noisy)

    for setting in ["op_setting_1", "op_setting_2", "op_setting_3"]:
        base = float(row[setting])
        row[setting] = float(base + random.gauss(0.0, abs(base) * noise_scale * 0.25 + 1e-5))

    return row


def run() -> None:
    args = parse_args()
    random.seed(args.seed)
    template = _load_template(args.seed_csv)

    if args.digital_twin_root.exists():
        print(f"[digital-twin] reference repo detected at: {args.digital_twin_root}")

    client = mqtt_client.Client(client_id=args.client_id, protocol=mqtt_client.MQTTv311)
    if args.username:
        client.username_pw_set(args.username, args.password)
    client.connect(args.broker, args.port, keepalive=60)
    client.loop_start()

    try:
        for cycle in range(1, int(args.cycles) + 1):
            source_idx = (cycle - 1) % len(template)
            source = template.iloc[source_idx].to_dict()
            row = _next_row(
                template_row=source,
                cycle=cycle,
                engine_id=args.engine_id,
                noise_scale=float(args.noise_scale),
                degradation_rate=float(args.degradation_rate),
            )
            client.publish(args.topic, payload=json.dumps(row), qos=args.qos)
            print(f"[digital-twin] cycle={cycle} published")
            time.sleep(max(float(args.interval_sec), 0.0))
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    run()
