"""
run_pipeline.py
---------------
Entry point for the audio restoration pipeline.

Usage:
  python run_pipeline.py input/recording.wav
  python run_pipeline.py input/recording.wav -o output/clean.wav
  python run_pipeline.py input/recording.wav --no-hf --no-denoise
  python run_pipeline.py input/recording.wav --threshold 5.0
  python run_pipeline.py input/recording.wav --save-intermediates
  python run_pipeline.py --demo
"""

import argparse
import logging
from pathlib import Path


from pipeline_runner import AudioRestorationPipeline
from audio_io import generate_test_signal, save_wav


def build_parser():
    parser = argparse.ArgumentParser(
        description="Multi-stage audio signal restoration pipeline"
    )
    parser.add_argument("input", nargs="?", help="Input WAV file")
    parser.add_argument("-o", "--output", default="output/restored.wav")
    parser.add_argument("--output-dir", default="output")

    parser.add_argument("--threshold", type=float, default=8.0,
                        help="Click detection threshold (lower = more sensitive)")
    parser.add_argument("--cutoff", type=float, default=15000.0,
                        help="HF filter cutoff frequency in Hz")

    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a synthetic test signal and run the pipeline on it")
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.demo:
        print("Demo mode: generating synthetic corrupted audio...")
        Path("demo_input").mkdir(exist_ok=True)
        _, corrupted = generate_test_signal(duration=8.0, sample_rate=44100,
                                             click_count=25, noise_level=0.03)
        demo_path = "demo_input/synthetic_corrupted.wav"
        save_wav(demo_path, corrupted, 44100)
        print(f"Input saved to: {demo_path}")
        args.input = demo_path
        if args.output == "output/restored.wav":
            args.output = "output/demo_restored.wav"

    if not args.input:
        parser.error("Provide an input WAV file or use --demo")

    pipeline = AudioRestorationPipeline(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.output_dir,
        run_denoising=not args.no_denoise,
        run_hf_filter=not args.no_hf,
        generate_plots=not args.no_plots,
        save_intermediates=args.save_intermediates,
        threshold=args.threshold,
        cutoff_hz=args.cutoff,
    )

    report = pipeline.run()
    pipeline.print_report(report)

    if report["plots"]:
        print("Plots saved:")
        for p in report["plots"]:
            print(f"  {p}")


if __name__ == "__main__":
    main()
