"""
utils/visualiser.py
-------------------
Generates diagnostic plots for the audio restoration pipeline.

Saves everything to disk (non-interactive) so the pipeline can run
headlessly in any environment.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft

logger = logging.getLogger(__name__)

COLORS = {
    "original": "#4C72B0",
    "repaired": "#55A868",
    "click":    "#C44E52",
    "score":    "#8172B2",
    "threshold":"#CCB974",
    "final":    "#64B5CD",
}


class PipelineVisualiser:
    def __init__(self, output_dir="output/plots", dpi=150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_waveform(self, original, repaired, sample_rate, filename="waveform.png"):
        t = np.arange(len(original)) / sample_rate
        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
        fig.suptitle("Waveform — Before and After Click Repair", fontsize=13)

        axes[0].plot(t, original, color=COLORS["original"], linewidth=0.4)
        axes[0].set_title("Original (corrupted)")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(t, repaired, color=COLORS["repaired"], linewidth=0.4)
        axes[1].set_title("After click repair")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_xlabel("Time (s)")

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_click_detection(self, signal, score, mask, threshold, sample_rate,
                             filename="click_detection.png"):
        t = np.arange(len(signal)) / sample_rate
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        fig.suptitle("Click Detection", fontsize=13)

        axes[0].plot(t, signal, color=COLORS["original"], linewidth=0.4, label="Signal")
        click_times = t[mask]
        axes[0].scatter(click_times, signal[mask], color=COLORS["click"],
                        s=12, zorder=5, label=f"Clicks ({mask.sum()})")
        axes[0].set_title("Signal with detected clicks")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(fontsize=8)

        axes[1].plot(t, score, color=COLORS["score"], linewidth=0.5, label="Score")
        axes[1].axhline(threshold, color=COLORS["threshold"], linewidth=1.2,
                        linestyle="--", label=f"Threshold ({threshold:.3f})")
        axes[1].set_title("Detection score")
        axes[1].set_ylabel("Score")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_spectrogram_comparison(self, before, after, sample_rate,
                                    filename="spectrogram.png"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Spectrogram — Before vs After Denoising", fontsize=13)

        for ax, sig, title in zip(axes, [before, after], ["Before", "After"]):
            _, _, Zxx = stft(sig, fs=sample_rate, nperseg=2048, noverlap=1536)
            power_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
            im = ax.imshow(
                power_db, origin="lower", aspect="auto",
                vmin=-80, vmax=0, cmap="magma",
                extent=[0, len(sig) / sample_rate, 0, sample_rate / 2 / 1000],
            )
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (kHz)")
            fig.colorbar(im, ax=ax, label="dB")

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_spectrum_comparison(self, original, final, sample_rate,
                                 filename="spectrum.png"):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Frequency Spectrum — Before vs After", fontsize=13)

        for sig, label, color in [(original, "Original", COLORS["original"]),
                                   (final, "Restored", COLORS["final"])]:
            freqs, _, Zxx = stft(sig, fs=sample_rate, nperseg=2048, noverlap=1536)
            avg_db = 20 * np.log10(np.abs(Zxx).mean(axis=1) + 1e-12)
            ax.plot(freqs / 1000, avg_db, label=label, color=color, linewidth=1.2, alpha=0.85)

        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, sample_rate / 2 / 1000)
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_full_summary(self, original, repaired, denoised, final,
                          score, mask, threshold, sample_rate,
                          filename="summary.png"):
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Audio Restoration Pipeline — Summary", fontsize=14)
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
        t   = np.arange(len(original)) / sample_rate

        # Original waveform with click markers
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(t, original, color=COLORS["original"], linewidth=0.3)
        click_idx = np.where(mask)[0]
        ax00.scatter(t[click_idx], original[click_idx],
                     color=COLORS["click"], s=8, zorder=5, label=f"{len(click_idx)} clicks")
        ax00.set_title("Original (corrupted)")
        ax00.set_ylabel("Amplitude")
        ax00.legend(fontsize=7)

        # Detection score
        ax01 = fig.add_subplot(gs[0, 1])
        ax01.plot(t, score, color=COLORS["score"], linewidth=0.4)
        ax01.axhline(threshold, color=COLORS["threshold"], linewidth=1,
                     linestyle="--", label=f"Threshold ({threshold:.3f})")
        ax01.set_title("Click detection score")
        ax01.set_ylabel("Score")
        ax01.legend(fontsize=7)

        # After repair
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.plot(t, repaired, color=COLORS["repaired"], linewidth=0.3)
        ax10.set_title("After click repair")
        ax10.set_ylabel("Amplitude")

        # Original spectrogram
        ax11 = fig.add_subplot(gs[1, 1])
        _, _, Zxx = stft(original, fs=sample_rate, nperseg=2048, noverlap=1536)
        power_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
        im = ax11.imshow(power_db, origin="lower", aspect="auto",
                         vmin=-80, vmax=0, cmap="magma",
                         extent=[0, len(original) / sample_rate, 0, sample_rate / 2 / 1000])
        ax11.set_title("Spectrogram (original)")
        ax11.set_xlabel("Time (s)")
        ax11.set_ylabel("Frequency (kHz)")
        fig.colorbar(im, ax=ax11, label="dB", fraction=0.046)

        # Final waveform
        ax20 = fig.add_subplot(gs[2, 0])
        ax20.plot(t, final, color=COLORS["final"], linewidth=0.3)
        ax20.set_title("Final output")
        ax20.set_ylabel("Amplitude")
        ax20.set_xlabel("Time (s)")

        # Spectrum comparison
        ax21 = fig.add_subplot(gs[2, 1])
        for sig, label, color in [(original, "Original", COLORS["original"]),
                                   (final, "Restored", COLORS["final"])]:
            freqs, _, Zxx2 = stft(sig, fs=sample_rate, nperseg=2048, noverlap=1536)
            avg_db = 20 * np.log10(np.abs(Zxx2).mean(axis=1) + 1e-12)
            ax21.plot(freqs / 1000, avg_db, label=label, color=color, linewidth=1.0, alpha=0.85)
        ax21.set_title("Spectrum before vs after")
        ax21.set_xlabel("Frequency (kHz)")
        ax21.set_ylabel("Magnitude (dB)")
        ax21.legend(fontsize=8)
        ax21.grid(True, alpha=0.25)
        ax21.set_xlim(0, sample_rate / 2 / 1000)

        return self._save(fig, filename)

    def _save(self, fig, filename):
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved plot: %s", path)
        return path
