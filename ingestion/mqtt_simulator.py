from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from pathlib import Path

import pandas as pd
from paho.mqtt import client as mqtt_client

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish C-MAPSS rows as simulated real-time MQTT stream.")
    parser.add_argument("--csv", type=Path, required=True, help="Scenario CSV file to stream.")
    parser.add_argument("--broker", required=True)
    parser.add_argument("--port", type=int, default=8883)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--client-id", default="rul-simulator")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--ca-cert", type=Path, required=True)
    parser.add_argument("--client-cert", type=Path)
    parser.add_argument("--client-key", type=Path)
    parser.add_argument("--delay-sec", type=float, default=0.20, help="Delay between row publishes.")
    parser.add_argument("--qos", type=int, default=1, choices=[0, 1, 2])
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    rows = df.to_dict(orient="records")

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

    client.connect(args.broker, args.port, keepalive=60)
    client.loop_start()
    try:
        for idx, row in enumerate(rows, start=1):
            payload = json.dumps(row)
            client.publish(args.topic, payload=payload, qos=args.qos)
            print(f"[sim] published {idx}/{len(rows)}")
            time.sleep(max(args.delay_sec, 0.0))
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    run()
