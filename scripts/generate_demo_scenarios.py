from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = REPO_ROOT / "examples" / "sample_cmapss_engine.csv"
OUT_DIR = REPO_ROOT / "examples" / "scenarios"
SEED = 42


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("s_")]


def create_stable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sensors = _sensor_columns(out)
    # Smooth most sensor variation to emulate highly stable operation.
    for col in sensors:
        out[col] = out[col].rolling(window=5, min_periods=1).mean()
    return out


def create_noisy(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    sensors = _sensor_columns(out)
    for col in sensors:
        std = float(out[col].std())
        noise = rng.normal(0.0, std * 0.08, size=len(out))
        out[col] = out[col] + noise
    return out


def create_rapid_degradation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sensors = _sensor_columns(out)
    start_idx = int(len(out) * 0.6)
    late = out.index >= out.index[start_idx]
    ramp = np.linspace(0.0, 1.0, late.sum())
    direction = {
        "s_2": 1.0,
        "s_3": 1.0,
        "s_4": 1.0,
        "s_7": 1.0,
        "s_11": -1.0,
        "s_12": 1.0,
        "s_15": -1.0,
        "s_20": -1.0,
    }
    for col in sensors:
        if col not in direction:
            continue
        span = float(out[col].max() - out[col].min())
        drift = direction[col] * 0.25 * span * ramp
        out.loc[late, col] = out.loc[late, col].to_numpy(dtype=float) + drift
    return out


def main() -> None:
    df = pd.read_csv(SOURCE_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    stable = create_stable(df)
    noisy = create_noisy(df, rng)
    rapid = create_rapid_degradation(df)

    stable_path = OUT_DIR / "scenario_stable_behavior.csv"
    noisy_path = OUT_DIR / "scenario_noisy_behavior.csv"
    rapid_path = OUT_DIR / "scenario_rapid_degradation.csv"

    stable.to_csv(stable_path, index=False)
    noisy.to_csv(noisy_path, index=False)
    rapid.to_csv(rapid_path, index=False)

    print("Demo scenarios generated:")
    print(f"- {stable_path}")
    print(f"- {noisy_path}")
    print(f"- {rapid_path}")


if __name__ == "__main__":
    main()
