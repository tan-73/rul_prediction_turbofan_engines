from __future__ import annotations

import json
from pathlib import Path

from backend.model_service import ModelService


def test_golden_inference_outputs_within_tolerance() -> None:
    root = Path(__file__).resolve().parents[1]
    golden_path = root / "reproducibility" / "golden_inference_outputs.json"
    assert golden_path.exists(), "Missing golden regression file. Run scripts/run_validation_suite.py first."

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    service = ModelService()

    max_rul_delta = 1.0
    max_ri_delta = 0.08

    for fixture, mode_map in golden.items():
        payload = (root / fixture).read_bytes()
        for mode, expected in mode_map.items():
            actual = service.infer(payload, model_mode=mode)

            actual_rul = float(actual["overall_mean_rul"])
            actual_ri = float(actual["overall_reliability_index"])
            exp_rul = float(expected["overall_mean_rul"])
            exp_ri = float(expected["overall_reliability_index"])

            assert abs(actual_rul - exp_rul) <= max_rul_delta, (
                f"{fixture} | {mode}: overall_mean_rul drifted. expected={exp_rul}, actual={actual_rul}"
            )
            assert abs(actual_ri - exp_ri) <= max_ri_delta, (
                f"{fixture} | {mode}: overall_reliability_index drifted. expected={exp_ri}, actual={actual_ri}"
            )
