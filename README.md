# signal_processing

A multi-stage pipeline for detecting and repairing corruption in audio files. The project covers the full restoration process — from automated click detection through spectral noise reduction and high-frequency roll-off — and produces a clean WAV file along with diagnostic plots at each stage.

The pipeline was originally developed in MATLAB. The core signal processing logic is being re-implemented here using NumPy and SciPy.

---

## How it works

```
Input WAV
│
▼
[1] Click Detection       median-filter residual + curvature → weighted MAD score → boolean mask
│
▼
[2] Click Repair          short regions: linear interpolation
longer regions: local median fill + Hann fade at boundaries
│
▼
[3] Spectral Denoising    STFT → per-frequency noise floor → sigmoid gain mask → inverse STFT
│
▼
[4] HF Roll-off           cosine-tapered gain envelope above a configurable cutoff frequency
│
▼
Output WAV + diagnostic plots + pipeline_report.json
```

Stages 3 and 4 are optional and can be skipped with flags.

---

## Project structure

```
signal_processing/
├── run_pipeline.py       # Entry point — run the pipeline from here
├── pipeline_runner.py    # Orchestrates all stages and writes the report
├── click_detector.py     # Click detection using MAD score
├── click_repairer.py     # Click repair — interpolation and median fill
├── spectral_denoiser.py  # STFT-based broadband noise reduction
├── hf_filter.py          # Cosine-tapered high-frequency roll-off
├── audio_io.py           # WAV loading/saving and synthetic signal generator
├── visualiser.py         # Matplotlib diagnostic figures
├── test_pipeline.py      # pytest suite
├── pipeline_config.json  # Default parameter values
├── input/                # Put your WAV files here
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting started

**Requirements:** Python 3.10 or newer

```bash
cd signal_processing

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Quick test — no audio file needed

```bash
python run_pipeline.py --demo
```

Generates a synthetic corrupted signal (composite sine wave with injected clicks and broadband noise), runs it through the full pipeline, and saves everything to `output/`.

### Process a real file

Drop your WAV file into the `input/` folder, then:

```bash
python run_pipeline.py input/recording.wav
```

### Specify an output path

```bash
python run_pipeline.py input/recording.wav -o output/recording_clean.wav
```

### All options

```bash
python run_pipeline.py input/recording.wav [options]

  -o, --output PATH          Output WAV path (default: output/restored.wav)
  --output-dir DIR           Directory for plots and report (default: output)
  --threshold FLOAT          Click detection threshold in MAD units (default: 8.0)
                             Lower = more sensitive, higher = fewer detections
  --cutoff FLOAT             HF filter cutoff frequency in Hz (default: 15000)
  --no-denoise               Skip the spectral denoising stage
  --no-hf                    Skip the HF roll-off filter
  --no-plots                 Skip plot generation
  --save-intermediates       Save a WAV file after each processing stage
  --demo                     Generate a synthetic signal and process it
  -v, --verbose              Show detailed logging output
```

---

## Output

After each run, the `output/` directory contains:
output/
├── restored.wav                 cleaned audio file
├── pipeline_report.json         run summary (click count, repair stats, timing)
└── plots/
├── waveform.png             before/after waveform with click positions marked
├── click_detection.png      detection score over time with threshold line
├── spectrogram.png          spectrogram before and after denoising
├── spectrum.png             average frequency spectrum before vs after
└── summary.png              all of the above in one figure

`pipeline_report.json` looks like this:

```json
{
  "input": "input/recording.wav",
  "output": "output/restored.wav",
  "sample_rate": 44100,
  "duration_s": 12.4,
  "clicks_found": 23,
  "short_repairs": 18,
  "medium_repairs": 5,
  "elapsed_s": 3.82,
  "plots": [...]
}
```

---

## Tuning

| Parameter | Default | When to adjust |
|-----------|---------|----------------|
| `--threshold` ↓ | 8.0 | Clicks are being missed |
| `--threshold` ↑ | 8.0 | Too many false positives on clean audio |
| `--cutoff` ↓ | 15000 Hz | High-frequency hiss starts below 15 kHz |
| `--no-denoise` | off | Denoising is introducing artefacts on musical content |
| `--no-hf` | off | High-frequency content is important and should not be attenuated |
| `--save-intermediates` | off | Useful for diagnosing which stage is causing a problem |

---

## Stage details

### Click Detection (`click_detector.py`)

Two signals are computed and combined into a single detection score:

- **Median-filter residual** — the signal is smoothed with a median filter; the difference between the original and the smoothed version is large where clicks are
- **Curvature** — the second difference of the signal, which highlights abrupt reversals in direction

Both are normalised using MAD (median absolute deviation) and blended with a configurable weight. Samples where the combined score exceeds the threshold are flagged as corrupted.

### Click Repair (`click_repairer.py`)

Corrupted regions are repaired differently depending on their length:

- **Short (≤ 8 samples)** — linear interpolation between the last clean sample before and after the region
- **Longer** — filled with the local median of nearby clean samples, with a raised-cosine fade at both edges to avoid introducing new discontinuities

### Spectral Denoising (`spectral_denoiser.py`)

1. Compute the STFT of the signal (2048-point FFT, Hann window)
2. Estimate a per-frequency noise floor from the first N frames
3. For each time-frequency bin, compute SNR relative to the noise floor
4. Map the SNR to a gain value using a sigmoid — bins well above the noise floor are preserved, bins near or below it are attenuated
5. Apply the gain and reconstruct with inverse STFT

Using a sigmoid instead of a hard mask avoids the "musical noise" artefact that hard spectral masking tends to produce.

### HF Roll-off (`hf_filter.py`)

Applies a raised-cosine (Hann) gain taper in the frequency domain between the cutoff frequency and a rolloff-end frequency. The transition is smooth rather than a hard cut, which eliminates the ringing artefacts associated with brick-wall filters.

---

## Running the tests

```bash
pytest test_pipeline.py -v
```

All tests use synthetically generated signals — no audio files needed.

---

## Ongoing work

- Edge-case validation for the MATLAB → NumPy/SciPy port
- Adaptive noise estimation that updates the noise floor estimate over time rather than using only the first N frames
- SNR improvement and spectral distortion metrics against a clean reference signal for quantitative evaluation
