"""
tests/test_pipeline.py
----------------------
Tests for the audio restoration pipeline stages.

All signals are generated synthetically — no real audio files needed.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from click_detector import ClickDetector
from click_repairer import ClickRepairer
from spectral_denoiser import SpectralDenoiser
from hf_filter import HFFilter
from audio_io import generate_test_signal, save_wav

SR = 44100


@pytest.fixture
def corrupted():
    clean, corrupted = generate_test_signal(duration=3.0, sample_rate=SR,
                                             click_count=15, seed=0)
    return clean, corrupted


class TestClickDetector:
    def test_returns_correct_shape(self, corrupted):
        _, signal = corrupted
        result = ClickDetector().detect(signal, SR)
        assert result["click_mask"].shape == signal.shape

    def test_finds_clicks(self, corrupted):
        _, signal = corrupted
        result = ClickDetector(threshold_mad=6.0).detect(signal, SR)
        assert result["click_count"] > 0

    def test_clean_signal_low_false_positives(self):
        t = np.linspace(0, 2.0, 2 * SR)
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = ClickDetector().detect(clean, SR)
        assert result["click_mask"].mean() < 0.005

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError):
            ClickDetector().detect(np.ones((100, 2)))


class TestClickRepairer:
    def test_output_shape_unchanged(self, corrupted):
        _, signal = corrupted
        mask = ClickDetector().detect(signal, SR)["click_mask"]
        repaired, _ = ClickRepairer().repair(signal, mask)
        assert repaired.shape == signal.shape

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            ClickRepairer().repair(np.ones(100), np.ones(50, dtype=bool))

    def test_short_repair_interpolates(self):
        signal = np.ones(100, dtype=np.float64)
        signal[50:53] = 5.0
        mask = np.zeros(100, dtype=bool)
        mask[50:53] = True
        repaired, info = ClickRepairer(short_threshold=8).repair(signal, mask)
        assert np.all(repaired[50:53] < 2.0)
        assert info["short"] == 1

    def test_repair_counts_correct(self, corrupted):
        _, signal = corrupted
        mask = ClickDetector().detect(signal, SR)["click_mask"]
        _, info = ClickRepairer().repair(signal, mask)
        assert info["short"] + info["medium"] == info["total"]


class TestSpectralDenoiser:
    def test_output_length_matches(self, corrupted):
        _, signal = corrupted
        output = SpectralDenoiser().denoise(signal, SR)
        assert len(output) == len(signal)

    def test_reduces_noise_energy(self):
        rng = np.random.default_rng(1)
        t   = np.linspace(0, 2.0, 2 * SR)
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        noisy = clean + rng.normal(0, 0.1, len(t))
        output = SpectralDenoiser(noise_frames=5, snr_pivot_db=3.0).denoise(noisy, SR)
        assert np.mean((output - clean) ** 2) < np.mean((noisy - clean) ** 2)


class TestHFFilter:
    def test_output_length_matches(self):
        t = np.linspace(0, 2.0, 2 * SR)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        output = HFFilter().apply(signal, SR)
        assert len(output) == len(signal)

    def test_attenuates_hf_content(self):
        t = np.linspace(0, 1.0, SR)
        signal = 0.4 * np.sin(2 * np.pi * 18000 * t)
        output = HFFilter(cutoff_hz=15000).apply(signal, SR)
        assert np.mean(output ** 2) < np.mean(signal ** 2) * 0.5

    def test_preserves_lf_content(self):
        t = np.linspace(0, 2.0, 2 * SR)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        output = HFFilter(cutoff_hz=15000).apply(signal, SR)
        assert np.mean(output ** 2) > np.mean(signal ** 2) * 0.95


class TestIntegration:
    def test_full_pipeline(self, tmp_path):
        from pipeline_runner import AudioRestorationPipeline
        
        _, signal = generate_test_signal(duration=2.0, sample_rate=SR, click_count=10)
        input_path  = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        save_wav(input_path, signal, SR)

        pipeline = AudioRestorationPipeline(
            input_path=input_path,
            output_path=output_path,
            output_dir=str(tmp_path / "out"),
            generate_plots=False,
        )
        report = pipeline.run()

        assert Path(output_path).exists()
        assert report["clicks_found"] >= 0
        assert report["elapsed_s"] > 0
