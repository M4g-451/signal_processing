"""
stages/click_detector.py
------------------------
Detects click-like corruption in audio signals.

Two signals are combined to catch clicks:
  1. Median filter residual — clicks show up as big spikes the filter can't predict
  2. Curvature (second derivative) — catches abrupt direction changes in the waveform

Both are normalised and blended into a single score. Samples above a
threshold (measured in MAD units) are marked as corrupted.
"""

import logging
import numpy as np
from scipy.signal import medfilt

logger = logging.getLogger(__name__)


class ClickDetector:
    """
    Detects impulsive click noise in a mono audio array.

    Args:
        median_window: Window size for the median filter (must be odd)
        residual_weight: How much to weight the residual vs curvature (0-1)
        threshold_mad: Detection threshold in MAD units. Lower = more sensitive.
        min_gap: Minimum sample gap between detected clicks
    """

    def __init__(self, median_window=11, residual_weight=0.6,
                 threshold_mad=8.0, min_gap=5):
        self.median_window = median_window if median_window % 2 == 1 else median_window + 1
        self.residual_weight = residual_weight
        self.threshold_mad = threshold_mad
        self.min_gap = min_gap

    def detect(self, signal, sample_rate=44100):
        """
        Run click detection on a mono signal.

        Returns a dict with:
            click_mask  — boolean array, True = corrupted sample
            score       — raw detection score
            threshold   — threshold value used
            click_count — number of clicks found
        """
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

        signal = signal.astype(np.float64)

        # Median filter residual: difference between signal and its smoothed version
        smooth = medfilt(signal, kernel_size=self.median_window)
        residual = np.abs(signal - smooth)

        # Curvature: second difference highlights abrupt direction changes
        curvature = np.abs(np.diff(signal, n=2, prepend=signal[:1], append=signal[-1:]))

        # Normalise both to comparable scale using MAD
        residual_norm  = self._mad_normalise(residual)
        curvature_norm = self._mad_normalise(curvature)

        # Blend the two scores
        w = np.clip(self.residual_weight, 0.0, 1.0)
        score = w * residual_norm + (1.0 - w) * curvature_norm

        # Threshold
        threshold = np.median(score) + self.threshold_mad * self._mad(score)
        mask = score > threshold

        # Suppress detections that are too close together
        mask = self._apply_min_gap(mask)

        click_count = int(mask.sum())
        logger.info(
            "Detected %d click(s) in %.2fs of audio",
            click_count, len(signal) / sample_rate
        )

        return {
            "click_mask":  mask,
            "score":       score,
            "threshold":   threshold,
            "click_count": click_count,
        }

    @staticmethod
    def _mad(x):
        """Median absolute deviation."""
        return float(np.median(np.abs(x - np.median(x)))) + 1e-12

    @staticmethod
    def _mad_normalise(x):
        med = np.median(x)
        mad = float(np.median(np.abs(x - med))) + 1e-12
        return (x - med) / mad

    def _apply_min_gap(self, mask):
        """Suppress clicks that are too close to each other."""
        if self.min_gap <= 1:
            return mask
        result = mask.copy()
        indices = np.where(mask)[0]
        for i in range(len(indices) - 1):
            if indices[i + 1] - indices[i] < self.min_gap:
                result[indices[i]] = False
        return result
