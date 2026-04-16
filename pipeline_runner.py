"""
pipeline/runner.py
------------------
Runs the full audio restoration pipeline from start to finish.

Order:
  1. Load the WAV file
  2. Detect clicks
  3. Repair clicks
  4. Spectral denoising (optional)
  5. HF roll-off filter (optional)
  6. Save the result
  7. Generate plots (optional)
"""

import json
import logging
import time
from pathlib import Path

from click_detector   import ClickDetector
from click_repairer   import ClickRepairer
from spectral_denoiser import SpectralDenoiser
from hf_filter        import HFFilter
from audio_io          import load_wav, save_wav
from visualiser        import PipelineVisualiser

logger = logging.getLogger(__name__)


class AudioRestorationPipeline:
    """
    Ties all the processing stages together.

    Args:
        input_path:       Path to the input WAV file
        output_path:      Where to save the restored WAV
        output_dir:       Directory for plots and the JSON report
        run_denoising:    Toggle spectral denoising
        run_hf_filter:    Toggle HF roll-off
        generate_plots:   Toggle diagnostic plots
        save_intermediates: Save a WAV after each stage
    """

    def __init__(self, input_path, output_path="output/restored.wav",
                 output_dir="output", run_denoising=True, run_hf_filter=True,
                 generate_plots=True, save_intermediates=False,
                 threshold=8.0, cutoff_hz=15000.0):
        self.threshold = threshold
        self.cutoff_hz = cutoff_hz
        self.input_path        = input_path
        self.output_path       = output_path
        self.output_dir        = Path(output_dir)
        self.run_denoising     = run_denoising
        self.run_hf_filter     = run_hf_filter
        self.generate_plots    = generate_plots
        self.save_intermediates = save_intermediates

    def run(self):
        """Run the pipeline. Returns a report dict."""
        t0 = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load
        logger.info("Loading: %s", self.input_path)
        audio = load_wav(self.input_path)
        signal = audio["signal"]
        sr     = audio["sample_rate"]

        # 2. Click detection
        logger.info("Running click detection...")
        detector = ClickDetector()
        detection = ClickDetector(threshold_mad=self.threshold).detect(signal, sr)
        click_mask  = detection["click_mask"]
        click_count = detection["click_count"]

        # 3. Click repair
        logger.info("Repairing %d click(s)...", click_count)
        repairer = ClickRepairer()
        repaired, repair_info = repairer.repair(signal, click_mask)

        if self.save_intermediates:
            save_wav(self.output_dir / "stage1_repaired.wav", repaired, sr)

        # 4. Spectral denoising
        denoised = repaired
        if self.run_denoising:
            logger.info("Running spectral denoising...")
            denoised = SpectralDenoiser().denoise(repaired, sr)
            if self.save_intermediates:
                save_wav(self.output_dir / "stage2_denoised.wav", denoised, sr)

        # 5. HF filter
        final = denoised
        if self.run_hf_filter:
            logger.info("Applying HF filter...")
            final = HFFilter(cutoff_hz=self.cutoff_hz).apply(denoised, sr)

        # 6. Save output
        save_wav(self.output_path, final, sr)
        logger.info("Saved: %s", self.output_path)

        # 7. Plots
        plots = []
        if self.generate_plots:
            logger.info("Generating plots...")
            plot_dir = self.output_dir / "plots"
            vis = PipelineVisualiser(output_dir=plot_dir)
            plots = [
                str(vis.plot_waveform(signal, repaired, sr)),
                str(vis.plot_click_detection(signal, detection["score"], click_mask, detection["threshold"], sr)),
                str(vis.plot_spectrogram_comparison(repaired, denoised, sr)),
                str(vis.plot_spectrum_comparison(signal, final, sr)),
                str(vis.plot_full_summary(signal, repaired, denoised, final,
                                          detection["score"], click_mask, detection["threshold"], sr)),
            ]

        elapsed = time.perf_counter() - t0

        report = {
            "input":          str(self.input_path),
            "output":         str(self.output_path),
            "sample_rate":    sr,
            "duration_s":     audio["duration"],
            "clicks_found":   click_count,
            "short_repairs":  repair_info["short"],
            "medium_repairs": repair_info["medium"],
            "elapsed_s":      round(elapsed, 2),
            "plots":          plots,
        }

        report_path = self.output_dir / "pipeline_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        return report

    def print_report(self, report):
        print("\n" + "=" * 48)
        print("  Audio Restoration — Report")
        print("=" * 48)
        print(f"  Input    : {report['input']}")
        print(f"  Output   : {report['output']}")
        print(f"  Duration : {report['duration_s']:.2f}s @ {report['sample_rate']} Hz")
        print(f"  Clicks   : {report['clicks_found']} found")
        print(f"  Repairs  : {report['short_repairs']} short, {report['medium_repairs']} medium")
        print(f"  Time     : {report['elapsed_s']}s")
        print("=" * 48 + "\n")
