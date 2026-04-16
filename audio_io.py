"""
utils/audio_io.py
-----------------
WAV file loading and saving.

load_wav normalises everything to mono float64 in [-1, 1].
save_wav converts back to 16-bit PCM.
generate_test_signal creates a synthetic corrupted signal for testing.
"""

import logging
from pathlib import Path

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def load_wav(path):
    """
    Load a WAV file and return a dict with:
        signal      — mono float64 array in [-1, 1]
        sample_rate — int
        duration    — float (seconds)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sample_rate, data = wavfile.read(str(path))

    # Stereo -> mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / max_val
    else:
        data = data.astype(np.float64)

    data = np.clip(data, -1.0, 1.0)
    duration = len(data) / sample_rate

    logger.info("Loaded %s (%.2fs, %d Hz)", path.name, duration, sample_rate)
    return {"signal": data, "sample_rate": sample_rate, "duration": duration}


def save_wav(path, signal, sample_rate):
    """Save a float64 signal as a 16-bit PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm)
    logger.info("Saved %s (%.2fs)", path.name, len(signal) / sample_rate)


def generate_test_signal(duration=5.0, sample_rate=44100, click_count=20,
                          noise_level=0.02, seed=42):
    """
    Generate a synthetic signal with injected clicks and noise for testing.

    Returns (clean, corrupted) as a tuple of float64 arrays.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n)

    # Simple composite tone
    clean = (
        0.40 * np.sin(2 * np.pi * 440 * t)
        + 0.25 * np.sin(2 * np.pi * 660 * t)
        + 0.15 * np.sin(2 * np.pi * 880 * t)
    )
    clean = clean / np.max(np.abs(clean))

    corrupted = clean.copy()
    corrupted += rng.normal(0, noise_level, n)

    # Inject clicks
    positions = rng.integers(1000, n - 1000, size=click_count)
    for pos in positions:
        width = rng.integers(1, 6)
        amp   = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
        corrupted[pos : pos + width] += amp

    corrupted = np.clip(corrupted, -1.0, 1.0)
    logger.info("Generated test signal: %.2fs, %d clicks injected", duration, click_count)
    return clean, corrupted
