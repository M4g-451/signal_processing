"""
stages/hf_filter.py
-------------------
High-frequency roll-off filter using a cosine taper.

Instead of a hard brick-wall cut (which causes ringing), the gain
gradually drops from 1 to 0 between the cutoff and rolloff_end
frequencies, following the shape of a raised cosine.
"""

import logging
import numpy as np
from scipy.signal import stft, istft

logger = logging.getLogger(__name__)


class HFFilter:
    """
    Args:
        cutoff_hz:     Frequency where attenuation starts
        rolloff_end_hz: Frequency where gain reaches zero (defaults to cutoff * 1.33)
        n_fft:         FFT size
        hop_length:    Hop size
    """

    def __init__(self, cutoff_hz=15000.0, rolloff_end_hz=None, n_fft=2048, hop_length=512):
        self.cutoff_hz     = cutoff_hz
        self.rolloff_end_hz = rolloff_end_hz
        self.n_fft         = n_fft
        self.hop_length    = hop_length

    def apply(self, signal, sample_rate=44100):
        """Apply HF roll-off and return the filtered signal."""
        signal = signal.astype(np.float64)
        rolloff_end = self.rolloff_end_hz or self.cutoff_hz * 1.33

        freqs, _, S = stft(signal, fs=sample_rate, window="hann",
                           nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)

        # Build the gain curve
        gain = np.ones(len(freqs))
        taper = (freqs > self.cutoff_hz) & (freqs <= rolloff_end)
        t = (freqs[taper] - self.cutoff_hz) / (rolloff_end - self.cutoff_hz)
        gain[taper] = 0.5 * (1.0 + np.cos(np.pi * t))
        gain[freqs > rolloff_end] = 0.0

        # Apply and reconstruct
        S_filtered = S * gain[:, np.newaxis]
        _, output = istft(S_filtered, fs=sample_rate, window="hann",
                          nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)

        if len(output) >= len(signal):
            output = output[:len(signal)]
        else:
            output = np.pad(output, (0, len(signal) - len(output)))

        logger.info("HF filter applied: cutoff=%.0fHz, rolloff_end=%.0fHz",
                    self.cutoff_hz, rolloff_end)
        return output
