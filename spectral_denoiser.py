"""
stages/spectral_denoiser.py
---------------------------
Broadband noise reduction using STFT spectral masking.

Steps:
1. Convert the signal to the frequency domain with STFT
2. Estimate a noise floor from the quietest frames
3. For each frequency bin, compute how far above the noise floor it is (SNR)
4. Apply a sigmoid gain — bins with high SNR are kept, noisy bins are reduced
5. Reconstruct the signal with inverse STFT

Using a sigmoid instead of a hard cutoff avoids the "musical noise" artefact
that hard masking tends to produce.
"""

import logging
import numpy as np
from scipy.signal import stft, istft

logger = logging.getLogger(__name__)


class SpectralDenoiser:
    """
    Args:
        n_fft:            FFT size (power of 2 recommended)
        hop_length:       Hop size between frames
        noise_frames:     Number of frames at the start used to estimate noise
        snr_pivot_db:     SNR at which gain = 0.5 (higher = more conservative)
        sigmoid_steepness: Controls sharpness of the gain transition
        min_gain:         Floor on the gain — never fully silence a bin
    """

    def __init__(self, n_fft=2048, hop_length=512, noise_frames=10,
                 snr_pivot_db=6.0, sigmoid_steepness=1.0, min_gain=0.02):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_frames = noise_frames
        self.snr_pivot_db = snr_pivot_db
        self.sigmoid_steepness = sigmoid_steepness
        self.min_gain = min_gain

    def denoise(self, signal, sample_rate=44100):
        """
        Denoise a mono audio signal. Returns the cleaned signal array.
        """
        signal = signal.astype(np.float64)

        # Forward STFT
        _, _, S = stft(signal, fs=sample_rate, window="hann",
                       nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        magnitude = np.abs(S)
        phase     = np.angle(S)

        # Estimate noise floor from the first N frames
        n = min(self.noise_frames, magnitude.shape[1])
        noise_profile = magnitude[:, :n].mean(axis=1)

        # SNR in dB for each bin
        eps    = 1e-12
        snr_db = 20.0 * np.log10(magnitude / (noise_profile[:, np.newaxis] + eps) + eps)

        # Sigmoid gain
        x    = self.sigmoid_steepness * (snr_db - self.snr_pivot_db)
        gain = 1.0 / (1.0 + np.exp(-x))
        gain = np.maximum(gain, self.min_gain)

        # Reconstruct
        S_clean = gain * magnitude * np.exp(1j * phase)
        _, output = istft(S_clean, fs=sample_rate, window="hann",
                          nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)

        # Match original length
        if len(output) >= len(signal):
            output = output[:len(signal)]
        else:
            output = np.pad(output, (0, len(signal) - len(output)))

        logger.info("Spectral denoising applied (%d freq bins)", S.shape[0])
        return output
