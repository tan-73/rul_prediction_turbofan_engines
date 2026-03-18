"""Node-RED MQTT subscriber for Streamlit integration.

Subscribes to Node-RED digital twin MQTT topics and writes state files
that Streamlit can read for live dashboard display.

Topics:
  - engine/rul       → RUL predictions, gate, CPC, health
  - engine/sensors/all → Full sensor payload

Usage:
  python ingestion/nodered_subscriber.py --broker 127.0.0.1 --port 1883
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LOGS_DIR = ROOT_DIR / "logs"
STATE_FILE = LOGS_DIR / "nodered_live_state.json"
EVENTS_FILE = LOGS_DIR / "nodered_events.ndjson"

TOPICS = ["engine/rul", "engine/sensors/all"]


def _write_state(state: dict) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _append_event(topic: str, payload: dict) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        **payload,
    }
    with open(EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


class NodeREDSubscriber:
    def __init__(self, broker: str = "127.0.0.1", port: int = 1883) -> None:
        self.broker = broker
        self.port = port
        self._state: dict = {}
        self._client = None
        self._running = False

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc == 0:
            print(f"[NodeRED-Sub] Connected to {self.broker}:{self.port}")
            for topic in TOPICS:
                client.subscribe(topic)
                print(f"[NodeRED-Sub] Subscribed to: {topic}")
        else:
            print(f"[NodeRED-Sub] Connection failed: rc={rc}")

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            payload = {"raw": msg.payload.decode("utf-8", errors="replace")}

        topic = msg.topic

        if topic == "engine/rul":
            self._state["latest_rul"] = payload
            self._state["rul_gate"] = payload.get("gate", "-")
            self._state["rul_value"] = payload.get("rul", 0)
            self._state["cpc"] = payload.get("cpc", 0)
            self._state["phys_risk"] = payload.get("physRisk", 0)
            self._state["cycle"] = payload.get("cycle", 0)
            self._state["health"] = payload.get("health", {})
            self._state["ci"] = payload.get("ci", [0, 0])

        elif topic == "engine/sensors/all":
            self._state["latest_sensors"] = payload

        _write_state(self._state)
        _append_event(topic, payload)

    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("[NodeRED-Sub] paho-mqtt not installed. Run: pip install paho-mqtt")
            return

        self._client = mqtt.Client(
            client_id="streamlit-nodered-sub",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.connect(self.broker, self.port, keepalive=60)
        self._running = True
        self._client.loop_start()

    def stop(self) -> None:
        if self._client:
            self._running = False
            self._client.loop_stop()
            self._client.disconnect()

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Node-RED MQTT subscriber for Streamlit")
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    sub = NodeREDSubscriber(broker=args.broker, port=args.port)
    sub.start()
    print("[NodeRED-Sub] Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sub.stop()
        print("[NodeRED-Sub] Stopped.")


if __name__ == "__main__":
    main()
