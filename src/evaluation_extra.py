"""Interval-forecast metrics."""
import numpy as np


def pinball(y_true, y_pred, q: float) -> float:
    """Pinball (quantile) loss; lower is better."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y_true, lo, hi) -> float:
    """Share of actual values that fall inside [lo, hi], in percent."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))) * 100)


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)))
