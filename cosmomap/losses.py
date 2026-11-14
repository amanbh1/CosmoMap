"""Training losses.

One configurable loss covers both models of the paper::

    L   = w_nll * NLL + w_mse * MSE + w_mae * MAE + w_huber * Huber + lambda_var * <s^2>
    NLL = 1/2 < (y - mu)^2 / s^2 + log s^2 >

* CosmoMap: the mean squared error of the predicted means (``w_mse = 1``, every
  other weight zero; the default).
* CosmoMap-sigma: the composite likelihood loss with weights 0.408, 0.150, 0.134,
  0.207 and 0.01, which also trains the per-voxel variance ``s^2``.

``y`` is the standardised target and the averages run over voxels and fields.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from cosmomap.config import Config

_LOG_VAR_CLIP = 10.0  # keep exp(+/- log_var) finite


def _huber(error: tf.Tensor, delta: float) -> tf.Tensor:
    abs_err = tf.abs(error)
    quad = tf.minimum(abs_err, delta)
    lin = abs_err - quad
    return 0.5 * tf.square(quad) + delta * lin


@tf.keras.utils.register_keras_serializable(package="cosmomap")
class CosmoMapLoss(keras.losses.Loss):
    """Weighted sum of point-estimate and likelihood terms.

    Parameters
    ----------
    n_out:
        Number of physical target channels (``y_true`` last-dim).
    heteroscedastic:
        If ``True``, ``y_pred`` has ``2 * n_out`` channels: ``[mean, log_var]``.
        The likelihood and variance terms need this and are skipped otherwise.
    """

    def __init__(
        self,
        n_out: int,
        *,
        heteroscedastic: bool = True,
        w_nll: float = 0.0,
        w_mse: float = 1.0,
        w_mae: float = 0.0,
        w_huber: float = 0.0,
        huber_delta: float = 1.0,
        lambda_variance: float = 0.0,
        name: str = "cosmomap_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.n_out = n_out
        self.heteroscedastic = heteroscedastic
        self.w_nll = w_nll
        self.w_mse = w_mse
        self.w_mae = w_mae
        self.w_huber = w_huber
        self.huber_delta = huber_delta
        self.lambda_variance = lambda_variance

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.heteroscedastic:
            mean = y_pred[..., : self.n_out]
            log_var = tf.clip_by_value(y_pred[..., self.n_out :], -_LOG_VAR_CLIP, _LOG_VAR_CLIP)
        else:
            mean, log_var = y_pred, None

        error = y_true - mean
        loss = self.w_mse * tf.reduce_mean(tf.square(error))
        if self.w_mae:
            loss += self.w_mae * tf.reduce_mean(tf.abs(error))
        if self.w_huber:
            loss += self.w_huber * tf.reduce_mean(_huber(error, self.huber_delta))

        if log_var is not None:
            if self.w_nll:
                # Gaussian NLL: 0.5 * [ (y - mu)^2 / sigma^2 + log sigma^2 ]
                nll = 0.5 * tf.reduce_mean(tf.square(error) * tf.exp(-log_var) + log_var)
                loss += self.w_nll * nll
            if self.lambda_variance:
                loss += self.lambda_variance * tf.reduce_mean(tf.exp(log_var))

        return loss

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "n_out": self.n_out,
                "heteroscedastic": self.heteroscedastic,
                "w_nll": self.w_nll,
                "w_mse": self.w_mse,
                "w_mae": self.w_mae,
                "w_huber": self.w_huber,
                "huber_delta": self.huber_delta,
                "lambda_variance": self.lambda_variance,
            }
        )
        return config


def make_loss(config: Config) -> CosmoMapLoss:
    """Build the loss from a :class:`~cosmomap.config.Config`."""
    lc = config.loss
    return CosmoMapLoss(
        n_out=config.n_out,
        heteroscedastic=config.model.heteroscedastic,
        w_nll=lc.w_nll,
        w_mse=lc.w_mse,
        w_mae=lc.w_mae,
        w_huber=lc.w_huber,
        huber_delta=lc.huber_delta,
        lambda_variance=lc.lambda_variance,
    )
