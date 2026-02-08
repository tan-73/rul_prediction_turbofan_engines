from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple, Union
import io

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler


WINDOW_LENGTH = 30
SHIFT = 1
EARLY_RUL = 125
NUM_TEST_WINDOWS = 5

NUM_HIDDENS = 64
NUM_LAYERS = 2
ATTENTION_SIZE = 32
RANDOM_SEED = 42

FD001_WEIGHTS_PATH = (
    Path("notebooks")
    / "cmapss_notebooks"
    / "attention_based_RUL"
    / "saved_weights"
    / "FD001"
    / "FD001_early_rul_125_GRU_rmse_14_21"
)

EXPECTED_NAMED_COLUMNS = [
    "unit_nr",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"s_{i}" for i in range(1, 22)]
RAW_COLUMN_NAMES = EXPECTED_NAMED_COLUMNS.copy()

# Exact drop list used in the original attention-based notebook preprocessing.
COLUMNS_TO_BE_DROPPED = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]


def process_input_data_with_targets(
    input_data: np.ndarray,
    target_data: np.ndarray | None = None,
    window_length: int = 1,
    shift: int = 1,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    num_batches = int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches * window_length * num_features).reshape(
        num_batches, window_length, num_features
    )

    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[
                (0 + shift * batch) : (0 + shift * batch + window_length), :
            ]
        return output_data

    output_targets = np.repeat(np.nan, repeats=num_batches)
    for batch in range(num_batches):
        output_data[batch, :, :] = input_data[
            (0 + shift * batch) : (0 + shift * batch + window_length), :
        ]
        output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
    return output_data, output_targets


def process_test_data(
    test_data_for_an_engine: np.ndarray,
    window_length: int,
    shift: int,
    num_test_windows: int = 1,
) -> Tuple[np.ndarray, int]:
    max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(
            test_data_for_an_engine[-required_len:, :],
            target_data=None,
            window_length=window_length,
            shift=shift,
        )
        return batched_test_data_for_an_engine, max_num_test_batches

    required_len = (num_test_windows - 1) * shift + window_length
    batched_test_data_for_an_engine = process_input_data_with_targets(
        test_data_for_an_engine[-required_len:, :],
        target_data=None,
        window_length=window_length,
        shift=shift,
    )
    return batched_test_data_for_an_engine, num_test_windows


class AdditiveAttentionForSeq(tf.keras.layers.Layer):
    def __init__(self, attention_size: int, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.Dense(attention_size)
        self.last_attention_weights = None

    def call(self, state, encoder_outputs):
        flat_state = []
        for item in state:
            if isinstance(item, (list, tuple)):
                if len(item) == 0:
                    continue
                flat_state.append(item[0])
            else:
                flat_state.append(item)

        if not flat_state:
            raise ValueError("Decoder hidden state is empty.")

        seq_len = encoder_outputs.shape[1]
        averaged_state = tf.reduce_mean(tf.stack(flat_state, axis=1), axis=1)
        state_rep = tf.repeat(tf.expand_dims(averaged_state, axis=1), repeats=seq_len, axis=1)
        concat = tf.concat((state_rep, encoder_outputs), axis=-1)
        scores = tf.nn.tanh(self.attention(concat))
        attention_weights = tf.nn.softmax(tf.reduce_sum(scores, axis=-1), axis=-1)
        self.last_attention_weights = attention_weights
        return tf.matmul(tf.expand_dims(attention_weights, axis=1), encoder_outputs)


class Seq2SeqEncoder(tf.keras.layers.Layer):
    def __init__(self, num_hiddens: int, num_layers: int, dropout: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout) for _ in range(num_layers)]
            ),
            return_sequences=True,
            return_state=True,
        )

    def call(self, x, **kwargs):
        output = self.rnn(x, **kwargs)
        state = output[1:]
        return output[0], state


class Seq2SeqAttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, num_hiddens: int, num_layers: int, dropout: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout) for _ in range(num_layers)]
            ),
            return_sequences=True,
            return_state=True,
        )
        self.attention = AdditiveAttentionForSeq(attention_size=ATTENTION_SIZE)
        self.dense = tf.keras.layers.Dense(1)

    def init_state(self, enc_outputs):
        outputs, hidden_state = enc_outputs
        return outputs, hidden_state

    def call(self, dec_input, state, **kwargs):
        enc_outputs, enc_hidden_state = state
        context = self.attention(enc_hidden_state, enc_outputs)
        rnn_input = tf.concat((dec_input, context), axis=-1)
        rnn_output = self.rnn(rnn_input, initial_state=enc_hidden_state, **kwargs)
        output = self.dense(tf.squeeze(rnn_output[0], axis=1))
        return output, rnn_output[1:]


class EncoderDecoder(tf.keras.Model):
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqAttentionDecoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_x, dec_x, **kwargs):
        enc_outputs = self.encoder(enc_x, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_x, dec_state, **kwargs)


def _read_csv_with_fallbacks(csv_file: Union[str, Path, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    attempts = [
        {"header": "infer"},
        {"header": None},
        {"sep": r"\s+", "engine": "python", "header": None},
    ]

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            if hasattr(csv_file, "seek"):
                csv_file.seek(0)
            frame = pd.read_csv(csv_file, **kwargs)
            if frame.shape[1] > 1:
                return frame
        except Exception as exc:  # pragma: no cover - defensive for file parsing.
            last_error = exc

    raise ValueError("Could not parse the uploaded CSV file.") from last_error


def _standardize_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    lowered = [str(col).strip().lower() for col in df.columns]

    if set(EXPECTED_NAMED_COLUMNS).issubset(set(lowered)):
        rename_map = {old: new for old, new in zip(df.columns, lowered)}
        df = df.rename(columns=rename_map)
        ordered = df[EXPECTED_NAMED_COLUMNS].copy()
        ordered.columns = list(range(26))
        return ordered

    if df.shape[1] < 26:
        raise ValueError(
            "Input CSV must include either standard C-MAPSS named columns or at least 26 raw columns."
        )

    numeric_df = df.iloc[:, :26].copy()
    numeric_df.columns = list(range(26))
    return numeric_df


def read_input_dataframe(csv_file: Union[str, Path, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    raw_df = _read_csv_with_fallbacks(csv_file)
    standardized = _standardize_input_frame(raw_df)
    standardized = standardized.apply(pd.to_numeric, errors="coerce")
    if standardized.isna().any().any():
        raise ValueError("Input CSV has non-numeric values in required sensor columns.")
    standardized.columns = RAW_COLUMN_NAMES
    return standardized


def load_attention_model(weights_path: Union[str, Path] = FD001_WEIGHTS_PATH) -> EncoderDecoder:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    model_path = Path(weights_path)
    if not model_path.exists() and not model_path.with_suffix(".index").exists():
        raise FileNotFoundError(f"Pretrained weights not found at: {model_path}")

    encoder = Seq2SeqEncoder(num_hiddens=NUM_HIDDENS, num_layers=NUM_LAYERS)
    decoder = Seq2SeqAttentionDecoder(num_hiddens=NUM_HIDDENS, num_layers=NUM_LAYERS)
    net = EncoderDecoder(encoder, decoder)

    num_features = 26 - len(COLUMNS_TO_BE_DROPPED)
    dummy_enc = tf.zeros((1, WINDOW_LENGTH, num_features), dtype=tf.float32)
    dummy_dec = tf.zeros((1, 1, num_features), dtype=tf.float32)
    net(dummy_enc, dummy_dec, training=False)

    try:
        net.load_weights(str(model_path))
    except ValueError:
        checkpoint = tf.train.Checkpoint(model=net)
        status = checkpoint.restore(str(model_path))
        status.expect_partial()

    return net


def preprocess_csv_for_inference(
    csv_file: Union[str, Path, io.BytesIO, io.StringIO],
    window_length: int = WINDOW_LENGTH,
    shift: int = SHIFT,
    num_test_windows: int = NUM_TEST_WINDOWS,
) -> Tuple[tf.Tensor, List[int], List[int]]:
    raw_df = read_input_dataframe(csv_file)
    indexed_raw_df = raw_df.copy()
    indexed_raw_df.columns = list(range(26))

    unit_column = indexed_raw_df[0].astype(int)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(indexed_raw_df.drop(columns=COLUMNS_TO_BE_DROPPED))
    scaled_df = pd.DataFrame(np.c_[unit_column.values, scaled])

    processed_test_data = []
    num_test_windows_list: List[int] = []
    engine_ids: List[int] = []

    for engine_id in np.sort(scaled_df[0].unique()):
        temp_test_data = scaled_df[scaled_df[0] == engine_id].drop(columns=[0]).values
        if len(temp_test_data) < window_length:
            raise ValueError(
                f"Engine {int(engine_id)} has only {len(temp_test_data)} rows. At least {window_length} rows are required."
            )

        test_data_for_engine, num_windows = process_test_data(
            temp_test_data,
            window_length=window_length,
            shift=shift,
            num_test_windows=num_test_windows,
        )
        processed_test_data.append(test_data_for_engine)
        num_test_windows_list.append(num_windows)
        engine_ids.append(int(engine_id))

    model_input = np.concatenate(processed_test_data).astype(np.float32)
    return tf.convert_to_tensor(model_input), engine_ids, num_test_windows_list


def _predict_seq2seq(net: EncoderDecoder, batched_data: tf.Tensor) -> np.ndarray:
    enc_outputs = net.encoder(batched_data, training=False)
    dec_state = net.decoder.init_state(enc_outputs)
    dec_x = batched_data[:, -1, tf.newaxis]
    y_pred, _ = net.decoder(dec_x, dec_state, training=False)
    return tf.squeeze(y_pred).numpy()


def _predict_seq2seq_with_attention(
    net: EncoderDecoder, batched_data: tf.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    enc_outputs = net.encoder(batched_data, training=False)
    dec_state = net.decoder.init_state(enc_outputs)
    dec_x = batched_data[:, -1, tf.newaxis]
    y_pred, _ = net.decoder(dec_x, dec_state, training=False)
    attention_weights = net.decoder.attention.last_attention_weights
    if attention_weights is None:
        raise ValueError("Attention weights were not produced by the model.")
    return tf.squeeze(y_pred).numpy(), attention_weights.numpy()


def _inverse_scale_rul(rul_pred_scaled: np.ndarray) -> np.ndarray:
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(np.arange(0, EARLY_RUL + 1).reshape(-1, 1))
    rul = target_scaler.inverse_transform(rul_pred_scaled.reshape(-1, 1)).reshape(-1)
    return np.clip(rul, 0.0, float(EARLY_RUL))


def predict_rul_from_csv(
    csv_file: Union[str, Path, io.BytesIO, io.StringIO],
    model: EncoderDecoder | None = None,
) -> Tuple[dict, float]:
    if model is None:
        model = load_attention_model()

    model_input, engine_ids, num_test_windows_list = preprocess_csv_for_inference(csv_file)
    rul_pred_scaled = np.atleast_1d(_predict_seq2seq(model, model_input))
    rul_pred = _inverse_scale_rul(rul_pred_scaled)

    preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
    mean_pred_for_each_engine = [float(np.mean(values)) for values in preds_for_each_engine]
    predictions = dict(zip(engine_ids, mean_pred_for_each_engine))
    overall_mean = float(np.mean(mean_pred_for_each_engine))
    return predictions, overall_mean


def predict_rul_detailed_from_csv(
    csv_file: Union[str, Path, io.BytesIO, io.StringIO],
    model: EncoderDecoder | None = None,
) -> Dict[str, object]:
    if model is None:
        model = load_attention_model()

    raw_df = read_input_dataframe(csv_file)
    model_input, engine_ids, num_test_windows_list = preprocess_csv_for_inference(csv_file)
    rul_pred_scaled, attention_weights = _predict_seq2seq_with_attention(model, model_input)
    rul_pred = _inverse_scale_rul(np.atleast_1d(rul_pred_scaled))

    preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
    attention_per_engine = np.split(attention_weights, np.cumsum(num_test_windows_list)[:-1])

    per_engine_mean = {int(engine_id): float(np.mean(values)) for engine_id, values in zip(engine_ids, preds_for_each_engine)}
    per_engine_windows = {int(engine_id): [float(v) for v in values] for engine_id, values in zip(engine_ids, preds_for_each_engine)}
    per_engine_attention_last = {
        int(engine_id): attention_values[-1].tolist()
        for engine_id, attention_values in zip(engine_ids, attention_per_engine)
    }

    return {
        "raw_df": raw_df,
        "engine_ids": [int(v) for v in engine_ids],
        "num_test_windows_list": num_test_windows_list,
        "per_engine_mean_rul": per_engine_mean,
        "per_engine_window_rul": per_engine_windows,
        "per_engine_last_attention": per_engine_attention_last,
        "overall_mean_rul": float(np.mean(list(per_engine_mean.values()))),
    }


def maintenance_status(predicted_rul: float, threshold: float = 40.0) -> str:
    if predicted_rul >= threshold:
        return "Healthy"
    return "Maintenance Required Soon"
