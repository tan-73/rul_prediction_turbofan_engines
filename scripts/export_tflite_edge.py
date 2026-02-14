from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Baseline/PI one-step attention model to TFLite.")
    parser.add_argument("--mode", default="Baseline", help="Model mode: Baseline or Physics-Informed.")
    parser.add_argument("--out-dir", type=Path, default=Path("edge_models"))
    parser.add_argument(
        "--quantization",
        choices=["none", "float16"],
        default="none",
        help="Optional TFLite quantization.",
    )
    return parser.parse_args()


def slugify_mode(mode: str) -> str:
    return mode.strip().lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    args = parse_args()

    import tensorflow as tf

    from inference.attention_model import (
        COLUMNS_TO_BE_DROPPED,
        WINDOW_LENGTH,
        get_model_weights_path,
        load_attention_model,
    )

    class OneStepAttentionModule(tf.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, WINDOW_LENGTH, None], dtype=tf.float32)])
        def infer(self, enc_x: tf.Tensor) -> tf.Tensor:
            enc_outputs = self.net.encoder(enc_x, training=False)
            dec_state = self.net.decoder.init_state(enc_outputs)
            dec_x = enc_x[:, -1, tf.newaxis]
            y_pred, _ = self.net.decoder(dec_x, dec_state, training=False)
            return tf.squeeze(y_pred, axis=-1)

    mode = args.mode
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_attention_model(get_model_weights_path(mode))
    wrapped = OneStepAttentionModule(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = Path(tmpdir) / f"{slugify_mode(mode)}_saved_model"
        tf.saved_model.save(wrapped, str(saved_model_dir), signatures={"serving_default": wrapped.infer})

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if args.quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

    mode_slug = slugify_mode(mode)
    quant_suffix = "fp16" if args.quantization == "float16" else "fp32"
    tflite_path = out_dir / f"{mode_slug}_one_step_{quant_suffix}.tflite"
    tflite_path.write_bytes(tflite_model)

    metadata = {
        "model_mode": mode,
        "weights_path": str(get_model_weights_path(mode)),
        "window_length": WINDOW_LENGTH,
        "num_features": int(26 - len(COLUMNS_TO_BE_DROPPED)),
        "quantization": args.quantization,
        "tflite_file": str(tflite_path),
    }
    metadata_path = out_dir / f"{mode_slug}_one_step_{quant_suffix}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("TFLite export complete.")
    print(f"- model: {tflite_path}")
    print(f"- metadata: {metadata_path}")


if __name__ == "__main__":
    main()
