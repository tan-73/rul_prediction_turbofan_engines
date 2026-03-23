"""Conditional VAE Trajectory Generator for RUL prediction.

Generates N probabilistic future RUL degradation trajectories given the
current engine state. This is the core generative AI component of the system.

Two modes:
  1. **cVAE mode** — uses a trained Conditional Variational Autoencoder
     to sample from the learned latent space of degradation patterns.
  2. **Monte Carlo fallback** — when no trained cVAE is available,
     uses the existing model predictions + uncertainty estimation
     to generate stochastic trajectory samples.

Architecture (cVAE):
  Encoder: sensor_window → μ, log_σ² (latent space)
  Decoder: z + condition → future RUL trajectory (T steps)
  Condition: current cycle position, mean sensor values, predicted RUL
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

REPO_ROOT = Path(__file__).resolve().parents[1]
CVAE_WEIGHTS_DIR = REPO_ROOT / "saved_models" / "cvae"

# Default generation parameters
DEFAULT_NUM_TRAJECTORIES = 50
DEFAULT_TRAJECTORY_LENGTH = 30
DEFAULT_LATENT_DIM = 16


# ═══════════════════════════════════════════════════════════════
# cVAE Model Architecture
# ═══════════════════════════════════════════════════════════════

class CVAEEncoder(tf.keras.layers.Layer):
    """Encodes sensor window + condition into latent distribution parameters."""

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(32, activation="relu")
        self.mu_layer = tf.keras.layers.Dense(latent_dim)
        self.log_var_layer = tf.keras.layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        return mu, log_var


class CVAEDecoder(tf.keras.layers.Layer):
    """Decodes latent vector + condition into future RUL trajectory."""

    def __init__(self, trajectory_length: int = DEFAULT_TRAJECTORY_LENGTH, **kwargs):
        super().__init__(**kwargs)
        self.trajectory_length = trajectory_length
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(128, activation="relu")
        self.output_layer = tf.keras.layers.Dense(trajectory_length)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)


class ConditionalVAE(tf.keras.Model):
    """Conditional Variational Autoencoder for RUL trajectory generation.

    Given an engine's current sensor state and operational context,
    generates multiple possible future degradation trajectories by
    sampling from the learned latent space.
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        trajectory_length: int = DEFAULT_TRAJECTORY_LENGTH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.trajectory_length = trajectory_length
        self.encoder = CVAEEncoder(latent_dim)
        self.decoder = CVAEDecoder(trajectory_length)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = μ + ε * σ."""
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def call(self, inputs, training=False):
        encoder_input, condition = inputs
        mu, log_var = self.encoder(encoder_input)
        z = self.reparameterize(mu, log_var)
        decoder_input = tf.concat([z, condition], axis=-1)
        reconstruction = self.decoder(decoder_input)
        return reconstruction, mu, log_var

    def generate_trajectories(
        self,
        condition: np.ndarray,
        n_samples: int = DEFAULT_NUM_TRAJECTORIES,
    ) -> np.ndarray:
        """Sample N trajectories from the prior conditioned on engine state."""
        condition_tensor = tf.constant(condition, dtype=tf.float32)
        if len(condition_tensor.shape) == 1:
            condition_tensor = tf.expand_dims(condition_tensor, 0)
        condition_batch = tf.repeat(condition_tensor, n_samples, axis=0)
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        decoder_input = tf.concat([z, condition_batch], axis=-1)
        trajectories = self.decoder(decoder_input, training=False)
        return trajectories.numpy()


def cvae_loss(reconstruction, target, mu, log_var, kl_weight: float = 0.5):
    """Combined reconstruction + KL divergence loss."""
    recon_loss = tf.reduce_mean(tf.square(reconstruction - target))
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return recon_loss + kl_weight * kl_loss


# ═══════════════════════════════════════════════════════════════
# Monte Carlo Trajectory Generator (fallback)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""
    n_trajectories: int = DEFAULT_NUM_TRAJECTORIES
    trajectory_length: int = DEFAULT_TRAJECTORY_LENGTH
    noise_scale: float = 0.08
    degradation_rate_mean: float = 1.0
    degradation_rate_std: float = 0.25
    physics_bound_upper: float = 125.0
    physics_bound_lower: float = 0.0
    monotonicity_strength: float = 0.85


def generate_mc_trajectories(
    current_rul: float,
    reliability_index: float,
    window_predictions: List[float],
    config: TrajectoryConfig | None = None,
) -> Dict[str, object]:
    """Generate Monte Carlo trajectory samples using prediction uncertainty.

    Uses the existing model prediction + window variance to create
    stochastic degradation paths. This serves as a fallback when
    no trained cVAE model is available.

    Args:
        current_rul: Current mean RUL prediction.
        reliability_index: Current RI score (affects trajectory variance).
        window_predictions: Recent window-level predictions for variance estimation.
        config: Trajectory generation parameters.

    Returns:
        Dict with trajectories array, statistics, and confidence intervals.
    """
    cfg = config or TrajectoryConfig()
    n = cfg.n_trajectories
    T = cfg.trajectory_length

    # Estimate prediction uncertainty from window variance
    window_std = float(np.std(window_predictions)) if len(window_predictions) > 1 else 2.0
    base_noise = max(window_std, 1.0) * cfg.noise_scale

    # Lower RI → higher uncertainty → wider trajectory fan
    uncertainty_scale = 1.0 + (1.0 - reliability_index) * 2.0

    # Sample degradation rates (how fast RUL declines per step)
    deg_rates = np.random.normal(
        cfg.degradation_rate_mean,
        cfg.degradation_rate_std * uncertainty_scale,
        size=n,
    )
    deg_rates = np.clip(deg_rates, 0.3, 3.0)

    # Generate trajectories
    trajectories = np.zeros((n, T))
    for i in range(n):
        rul = current_rul
        for t in range(T):
            noise = np.random.normal(0, base_noise * uncertainty_scale)
            rul_step = deg_rates[i] + noise
            if np.random.random() < cfg.monotonicity_strength:
                rul_step = abs(rul_step)
            rul = rul - rul_step
            rul = np.clip(rul, cfg.physics_bound_lower, cfg.physics_bound_upper)
            trajectories[i, t] = rul

    # Compute statistics across trajectories
    mean_trajectory = np.mean(trajectories, axis=0)
    std_trajectory = np.std(trajectories, axis=0)
    ci_lower = np.percentile(trajectories, 2.5, axis=0)
    ci_upper = np.percentile(trajectories, 97.5, axis=0)
    ci_25 = np.percentile(trajectories, 25, axis=0)
    ci_75 = np.percentile(trajectories, 75, axis=0)
    median_trajectory = np.median(trajectories, axis=0)

    # Time-to-threshold analysis
    failure_threshold = 10.0
    times_to_failure = []
    for i in range(n):
        below = np.where(trajectories[i] < failure_threshold)[0]
        if len(below) > 0:
            times_to_failure.append(int(below[0]))
        else:
            times_to_failure.append(T)

    mean_ttf = float(np.mean(times_to_failure))
    ttf_ci = (float(np.percentile(times_to_failure, 5)),
              float(np.percentile(times_to_failure, 95)))

    # Physics violation rate across trajectories
    phys_violations = float(np.mean(
        np.any((trajectories < cfg.physics_bound_lower) |
               (trajectories > cfg.physics_bound_upper), axis=1)
    ))

    return {
        "trajectories": trajectories.tolist(),
        "mean": mean_trajectory.tolist(),
        "median": median_trajectory.tolist(),
        "std": std_trajectory.tolist(),
        "ci_95_lower": ci_lower.tolist(),
        "ci_95_upper": ci_upper.tolist(),
        "ci_50_lower": ci_25.tolist(),
        "ci_50_upper": ci_75.tolist(),
        "n_trajectories": n,
        "trajectory_length": T,
        "current_rul": current_rul,
        "mean_time_to_failure": mean_ttf,
        "ttf_ci_90": list(ttf_ci),
        "physics_violation_rate": round(phys_violations, 4),
        "generation_mode": "monte_carlo",
    }


# ═══════════════════════════════════════════════════════════════
# Unified Trajectory Generator Interface
# ═══════════════════════════════════════════════════════════════

class TrajectoryGenerator:
    """Unified interface for trajectory generation.

    Attempts to use a trained cVAE model if available,
    otherwise falls back to Monte Carlo simulation.
    """

    def __init__(self, cvae_weights_dir: Path | str | None = None):
        self._cvae: Optional[ConditionalVAE] = None
        self._weights_dir = Path(cvae_weights_dir) if cvae_weights_dir else CVAE_WEIGHTS_DIR

        if HAS_TF and self._weights_dir.exists():
            try:
                self._load_cvae()
            except Exception:
                self._cvae = None

    def _load_cvae(self) -> None:
        """Attempt to load pre-trained cVAE weights."""
        model = ConditionalVAE()
        # Build model with dummy input
        dummy_enc = tf.zeros((1, 14))  # 14 sensor features
        dummy_cond = tf.zeros((1, 3))  # condition: rul, ri, cycle_frac
        model([dummy_enc, dummy_cond], training=False)

        weights_path = self._weights_dir / "cvae_weights"
        if Path(f"{weights_path}.index").exists():
            model.load_weights(str(weights_path))
            self._cvae = model

    @property
    def mode(self) -> str:
        return "cvae" if self._cvae is not None else "monte_carlo"

    def generate(
        self,
        current_rul: float,
        reliability_index: float,
        window_predictions: List[float],
        n_trajectories: int = DEFAULT_NUM_TRAJECTORIES,
        trajectory_length: int = DEFAULT_TRAJECTORY_LENGTH,
        sensor_means: Optional[np.ndarray] = None,
        current_cycle_fraction: float = 0.5,
    ) -> Dict[str, object]:
        """Generate trajectory samples.

        Args:
            current_rul: Current mean RUL prediction.
            reliability_index: Current RI score.
            window_predictions: Recent window-level predictions.
            n_trajectories: Number of trajectory samples.
            trajectory_length: Steps into the future.
            sensor_means: Mean sensor values (for cVAE conditioning).
            current_cycle_fraction: Current cycle / max cycle estimate.

        Returns:
            Dict with trajectories, statistics, and confidence intervals.
        """
        if self._cvae is not None and sensor_means is not None:
            return self._generate_cvae(
                current_rul, reliability_index, sensor_means,
                current_cycle_fraction, n_trajectories,
            )

        config = TrajectoryConfig(
            n_trajectories=n_trajectories,
            trajectory_length=trajectory_length,
        )
        return generate_mc_trajectories(
            current_rul, reliability_index, window_predictions, config,
        )

    def _generate_cvae(
        self,
        current_rul: float,
        reliability_index: float,
        sensor_means: np.ndarray,
        cycle_fraction: float,
        n_samples: int,
    ) -> Dict[str, object]:
        """Generate trajectories using trained cVAE."""
        condition = np.array([current_rul / 125.0, reliability_index, cycle_fraction],
                            dtype=np.float32)
        trajectories = self._cvae.generate_trajectories(condition, n_samples=n_samples)

        # Scale back to RUL domain
        trajectories = trajectories * 125.0
        trajectories = np.clip(trajectories, 0.0, 125.0)

        mean_traj = np.mean(trajectories, axis=0)
        ci_lower = np.percentile(trajectories, 2.5, axis=0)
        ci_upper = np.percentile(trajectories, 97.5, axis=0)

        return {
            "trajectories": trajectories.tolist(),
            "mean": mean_traj.tolist(),
            "median": np.median(trajectories, axis=0).tolist(),
            "std": np.std(trajectories, axis=0).tolist(),
            "ci_95_lower": ci_lower.tolist(),
            "ci_95_upper": ci_upper.tolist(),
            "ci_50_lower": np.percentile(trajectories, 25, axis=0).tolist(),
            "ci_50_upper": np.percentile(trajectories, 75, axis=0).tolist(),
            "n_trajectories": n_samples,
            "trajectory_length": trajectories.shape[1],
            "current_rul": current_rul,
            "generation_mode": "cvae",
        }
