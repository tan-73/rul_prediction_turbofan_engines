from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.attention_model import (
    COLUMNS_TO_BE_DROPPED,
    NUM_LAYERS,
    NUM_HIDDENS,
    RANDOM_SEED,
    EncoderDecoder,
    Seq2SeqAttentionDecoder,
    Seq2SeqEncoder,
    process_input_data_with_targets,
    process_test_data,
)


@dataclass
class TrainConfig:
    window_length: int = 30
    shift: int = 1
    early_rul: int = 125
    num_test_windows: int = 5
    val_engine_fraction: float = 0.2
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs_baseline: int = 12
    epochs_pi: int = 18
    patience: int = 6
    lambda_mono: float = 0.20
    lambda_smooth: float = 0.10
    lambda_bound: float = 0.05
    monotonic_delta: float = 0.01


def process_targets(data_length: int, early_rul: int) -> np.ndarray:
    early_rul_duration = data_length - early_rul
    if early_rul_duration <= 0:
        return np.arange(data_length - 1, -1, -1)
    return np.concatenate((np.full(early_rul_duration, early_rul), np.arange(early_rul - 1, -1, -1)))


def set_determinism(seed: int = RANDOM_SEED) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def load_cmapss(
    train_path: Path,
    test_path: Path,
    rul_path: Path,
    config: TrainConfig,
) -> Dict[str, np.ndarray]:
    train_data = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_data = pd.read_csv(test_path, sep=r"\s+", header=None)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)[0].values.astype(np.float32)

    train_first = train_data[0].values
    test_first = test_data[0].values

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.drop(columns=COLUMNS_TO_BE_DROPPED))
    test_scaled = scaler.transform(test_data.drop(columns=COLUMNS_TO_BE_DROPPED))

    train_df = pd.DataFrame(np.c_[train_first, train_scaled])
    test_df = pd.DataFrame(np.c_[test_first, test_scaled])

    processed_train_x: List[np.ndarray] = []
    processed_train_y: List[np.ndarray] = []
    train_engine_ids: List[np.ndarray] = []
    train_cycle_indices: List[np.ndarray] = []

    train_engines = np.sort(train_df[0].unique())
    for engine_id in train_engines:
        temp = train_df[train_df[0] == engine_id].drop(columns=[0]).values
        if len(temp) < config.window_length:
            continue
        targets = process_targets(data_length=len(temp), early_rul=config.early_rul)
        x, y = process_input_data_with_targets(
            temp,
            target_data=targets,
            window_length=config.window_length,
            shift=config.shift,
        )
        processed_train_x.append(x.astype(np.float32))
        processed_train_y.append(y.astype(np.float32))
        train_engine_ids.append(np.full(len(y), int(engine_id), dtype=np.int32))
        cycle_start = config.window_length - 1
        cycle_end = cycle_start + len(y)
        train_cycle_indices.append(np.arange(cycle_start, cycle_end, dtype=np.int32))

    x_train_all = np.concatenate(processed_train_x)
    y_train_all = np.concatenate(processed_train_y)
    train_engine_all = np.concatenate(train_engine_ids)
    train_cycle_all = np.concatenate(train_cycle_indices)

    rng = np.random.default_rng(RANDOM_SEED)
    unique_engines = np.array(sorted(np.unique(train_engine_all)))
    rng.shuffle(unique_engines)
    val_count = max(1, int(round(len(unique_engines) * config.val_engine_fraction)))
    val_engine_set = set(unique_engines[:val_count].tolist())

    val_mask = np.isin(train_engine_all, list(val_engine_set))
    tr_mask = ~val_mask

    x_tr = x_train_all[tr_mask]
    y_tr = y_train_all[tr_mask]
    e_tr = train_engine_all[tr_mask]
    c_tr = train_cycle_all[tr_mask]

    x_val = x_train_all[val_mask]
    y_val = y_train_all[val_mask]
    e_val = train_engine_all[val_mask]
    c_val = train_cycle_all[val_mask]

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_tr_s = target_scaler.fit_transform(y_tr.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = target_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    processed_test_data: List[np.ndarray] = []
    num_test_windows_list: List[int] = []
    test_engines = np.sort(test_df[0].unique())
    for engine_id in test_engines:
        temp = test_df[test_df[0] == engine_id].drop(columns=[0]).values
        if len(temp) < config.window_length:
            continue
        x_test_engine, num_windows = process_test_data(
            temp,
            window_length=config.window_length,
            shift=config.shift,
            num_test_windows=config.num_test_windows,
        )
        processed_test_data.append(x_test_engine.astype(np.float32))
        num_test_windows_list.append(int(num_windows))

    x_test = np.concatenate(processed_test_data).astype(np.float32)
    return {
        "x_tr": x_tr,
        "y_tr": y_tr_s,
        "engine_tr": e_tr,
        "cycle_tr": c_tr,
        "x_val": x_val,
        "y_val": y_val_s,
        "engine_val": e_val,
        "cycle_val": c_val,
        "x_test": x_test,
        "true_rul_test": true_rul,
        "num_test_windows_list": np.array(num_test_windows_list, dtype=np.int32),
        "target_scaler": target_scaler,
    }


def make_model(num_features: int) -> EncoderDecoder:
    encoder = Seq2SeqEncoder(num_hiddens=NUM_HIDDENS, num_layers=NUM_LAYERS)
    decoder = Seq2SeqAttentionDecoder(num_hiddens=NUM_HIDDENS, num_layers=NUM_LAYERS)
    model = EncoderDecoder(encoder=encoder, decoder=decoder)
    dummy_x = tf.zeros((1, 30, num_features), dtype=tf.float32)
    dummy_dec = tf.zeros((1, 1, num_features), dtype=tf.float32)
    model(dummy_x, dummy_dec, training=False)
    return model


def predict_scaled(model: EncoderDecoder, x: np.ndarray, batch_size: int) -> np.ndarray:
    ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    out: List[np.ndarray] = []
    for batch in ds:
        enc_outputs = model.encoder(batch, training=False)
        dec_state = model.decoder.init_state(enc_outputs)
        dec_x = batch[:, -1, tf.newaxis]
        y_pred, _ = model.decoder(dec_x, dec_state, training=False)
        out.append(tf.squeeze(y_pred, axis=-1).numpy())
    return np.concatenate(out, axis=0)


def monotonic_loss(y_pred: tf.Tensor, engine_ids: tf.Tensor, cycle_ids: tf.Tensor, delta: float) -> tf.Tensor:
    y = tf.reshape(y_pred, [-1])
    key = tf.cast(engine_ids, tf.int64) * tf.constant(1000000, dtype=tf.int64) + tf.cast(cycle_ids, tf.int64)
    order = tf.argsort(key)
    y_sorted = tf.gather(y, order)
    e_sorted = tf.gather(engine_ids, order)
    dy = y_sorted[1:] - y_sorted[:-1]
    same_engine = tf.cast(tf.equal(e_sorted[1:], e_sorted[:-1]), tf.float32)
    violation = tf.nn.relu(dy - float(delta)) * same_engine
    denom = tf.reduce_sum(same_engine) + 1e-6
    return tf.reduce_sum(violation) / denom


def smoothness_loss(y_pred: tf.Tensor, engine_ids: tf.Tensor, cycle_ids: tf.Tensor) -> tf.Tensor:
    y = tf.reshape(y_pred, [-1])
    key = tf.cast(engine_ids, tf.int64) * tf.constant(1000000, dtype=tf.int64) + tf.cast(cycle_ids, tf.int64)
    order = tf.argsort(key)
    y_sorted = tf.gather(y, order)
    e_sorted = tf.gather(engine_ids, order)
    if tf.size(y_sorted) < 3:
        return tf.constant(0.0, dtype=tf.float32)
    y0 = y_sorted[:-2]
    y1 = y_sorted[1:-1]
    y2 = y_sorted[2:]
    e0 = e_sorted[:-2]
    e1 = e_sorted[1:-1]
    e2 = e_sorted[2:]
    same_engine = tf.cast(tf.logical_and(tf.equal(e0, e1), tf.equal(e1, e2)), tf.float32)
    second = tf.abs(y2 - 2.0 * y1 + y0) * same_engine
    denom = tf.reduce_sum(same_engine) + 1e-6
    return tf.reduce_sum(second) / denom


def bound_loss(y_pred: tf.Tensor) -> tf.Tensor:
    y = tf.reshape(y_pred, [-1])
    return tf.reduce_mean(tf.nn.relu(-y) + tf.nn.relu(y - 1.0))


def evaluate_test(
    model: EncoderDecoder,
    x_test: np.ndarray,
    target_scaler: MinMaxScaler,
    num_test_windows_list: np.ndarray,
    true_rul_test: np.ndarray,
    batch_size: int,
) -> Dict[str, float]:
    pred_scaled = predict_scaled(model, x_test, batch_size=batch_size)
    pred_unscaled = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    pred_unscaled = np.clip(pred_unscaled, 0.0, 125.0)
    per_engine = np.split(pred_unscaled, np.cumsum(num_test_windows_list)[:-1])
    pred_mean = np.array([float(np.mean(v)) for v in per_engine], dtype=np.float32)
    rmse = float(math.sqrt(mean_squared_error(true_rul_test, pred_mean)))
    mae = float(mean_absolute_error(true_rul_test, pred_mean))
    diff = pred_mean - true_rul_test
    nasa_score = float(np.sum(np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)))
    return {"rmse": rmse, "mae": mae, "nasa_score": nasa_score}


def train(
    model: EncoderDecoder,
    data: Dict[str, np.ndarray],
    config: TrainConfig,
    out_dir: Path,
) -> Dict[str, float]:
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    mse = tf.keras.losses.MeanSquaredError()

    train_ds = tf.data.Dataset.from_tensor_slices(
        (data["x_tr"], data["y_tr"], data["engine_tr"], data["cycle_tr"])
    ).batch(config.batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(data["x_val"]).batch(config.batch_size)

    best_val_rmse = float("inf")
    best_weights = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    total_epochs = config.epochs_baseline + config.epochs_pi
    for epoch in range(total_epochs):
        use_pi = epoch >= config.epochs_baseline
        epoch_losses = []
        for x_batch, y_batch, e_batch, c_batch in train_ds:
            with tf.GradientTape() as tape:
                enc_outputs = model.encoder(x_batch, training=True)
                dec_state = model.decoder.init_state(enc_outputs)
                dec_x = x_batch[:, -1, tf.newaxis]
                y_pred, _ = model.decoder(dec_x, dec_state, training=True)
                y_pred_s = tf.squeeze(y_pred, axis=-1)
                l_data = mse(y_batch, y_pred_s)
                if use_pi:
                    l_mono = monotonic_loss(y_pred_s, e_batch, c_batch, delta=config.monotonic_delta)
                    l_smooth = smoothness_loss(y_pred_s, e_batch, c_batch)
                    l_bound = bound_loss(y_pred_s)
                    loss = (
                        l_data
                        + config.lambda_mono * l_mono
                        + config.lambda_smooth * l_smooth
                        + config.lambda_bound * l_bound
                    )
                else:
                    l_mono = tf.constant(0.0, dtype=tf.float32)
                    l_smooth = tf.constant(0.0, dtype=tf.float32)
                    l_bound = tf.constant(0.0, dtype=tf.float32)
                    loss = l_data

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(
                {
                    "loss": float(loss.numpy()),
                    "data_loss": float(l_data.numpy()),
                    "mono_loss": float(l_mono.numpy()),
                    "smooth_loss": float(l_smooth.numpy()),
                    "bound_loss": float(l_bound.numpy()),
                }
            )

        val_pred_scaled = predict_scaled(model, data["x_val"], batch_size=config.batch_size)
        val_pred = data["target_scaler"].inverse_transform(val_pred_scaled.reshape(-1, 1)).reshape(-1)
        val_true = data["target_scaler"].inverse_transform(data["y_val"].reshape(-1, 1)).reshape(-1)
        val_rmse = float(math.sqrt(mean_squared_error(val_true, np.clip(val_pred, 0.0, 125.0))))

        avg = {k: float(np.mean([x[k] for x in epoch_losses])) for k in epoch_losses[0]}
        avg["val_rmse"] = val_rmse
        avg["epoch"] = float(epoch + 1)
        avg["mode"] = 1.0 if use_pi else 0.0
        history.append(avg)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_weights is not None:
        model.set_weights(best_weights)

    metrics = evaluate_test(
        model=model,
        x_test=data["x_test"],
        target_scaler=data["target_scaler"],
        num_test_windows_list=data["num_test_windows_list"],
        true_rul_test=data["true_rul_test"],
        batch_size=config.batch_size,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "train_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "test_metrics.csv", index=False)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    ckpt_path = out_dir / "FD001_attention_physics_informed"
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save(str(ckpt_path))
    metrics["best_val_rmse"] = best_val_rmse
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline + physics-informed attention model on C-MAPSS FD001.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--rul-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("saved_models/cmapss/attn_pi_fd001"))
    parser.add_argument("--epochs-baseline", type=int, default=12)
    parser.add_argument("--epochs-pi", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-mono", type=float, default=0.20)
    parser.add_argument("--lambda-smooth", type=float, default=0.10)
    parser.add_argument("--lambda-bound", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_determinism()
    cfg = TrainConfig(
        epochs_baseline=args.epochs_baseline,
        epochs_pi=args.epochs_pi,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_mono=args.lambda_mono,
        lambda_smooth=args.lambda_smooth,
        lambda_bound=args.lambda_bound,
    )
    data = load_cmapss(args.train_path, args.test_path, args.rul_path, cfg)
    model = make_model(num_features=data["x_tr"].shape[-1])
    metrics = train(model, data, cfg, args.out_dir)
    print("Training complete.")
    for key, value in metrics.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
