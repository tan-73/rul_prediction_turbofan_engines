from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal-first inference runner (Raspberry Pi friendly).")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV for one or more engines.")
    parser.add_argument("--mode", default="Baseline", help="Model mode: Baseline or Physics-Informed.")
    parser.add_argument(
        "--backend",
        default="attention",
        help="Model backend: attention (default) or artifact.",
    )
    parser.add_argument("--compare", action="store_true", help="Run side-by-side Baseline vs PI output.")
    parser.add_argument("--replay-engine", type=int, help="Optional engine id for streaming replay.")
    parser.add_argument("--replay-step", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from backend.model_service import ModelService

    service = ModelService()
    csv_bytes = args.csv.read_bytes()

    if args.compare:
        result = service.compare(csv_bytes, model_backend=args.backend)
    elif args.replay_engine is not None:
        result = service.replay(
            csv_bytes,
            model_mode=args.mode,
            engine_id=int(args.replay_engine),
            step=max(int(args.replay_step), 1),
            model_backend=args.backend,
        )
    else:
        result = service.infer(csv_bytes, model_mode=args.mode, model_backend=args.backend)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
