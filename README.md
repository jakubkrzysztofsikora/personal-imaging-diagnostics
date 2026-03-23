<h1 align="center">Medical Imaging Analysis</h1>

<p align="center">
  Analyze medical images locally and privately using AI — no data ever leaves your machine.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/built%20with-Streamlit-ff4b4b" alt="Built with Streamlit">
</p>

---

> **Disclaimer:** This tool is for **educational and informational purposes only**. It does not provide medical diagnoses. Always consult a qualified healthcare professional for medical advice.

## What It Does

Upload a medical image — X-ray, CT, or MRI — and get a structured AI analysis in four steps:

1. **Modality Identification** — Determines the imaging type
2. **Systematic Observations** — Notes findings and abnormalities
3. **Diagnostic Synthesis** — Provides provisional assessments with confidence scores
4. **Referral Recommendations** — Suggests relevant specialist consultations

All processing runs **100% locally** using [Ollama](https://ollama.com/) or [mlx-lm](https://github.com/ml-explore/mlx-lm). No external API calls. No cloud uploads. Your images stay on your machine.

## Features

- **DICOM, PNG, and JPEG support** — Full DICOM parsing with automatic HU calibration, windowing, and photometric inversion
- **Privacy-first design** — Zero network calls for inference; everything runs locally
- **Governor pattern** — AI findings are always provisional and gated behind a disclaimer acknowledgment
- **Model management** — Pull, list, and delete Ollama models directly from the UI
- **Desktop mode** — Run as a native desktop window (via pywebview)

## Quick Start

### 1. Install Ollama

Download and install from [ollama.com](https://ollama.com/), then pull a vision model:

```bash
ollama pull llama3.2-vision
```

Make sure Ollama is running (`ollama serve` or launch the desktop app).

> **Apple Silicon alternative:** You can use [mlx-lm](https://github.com/ml-explore/mlx-lm) instead of Ollama — install it with `pip install mlx-lm`. The app downloads the model on first use.

### 2. Set up the app

```bash
git clone https://github.com/jakubkrzysztofsikora/personal-imaging-diagnostics.git
cd personal-imaging-diagnostics

python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 3. Run it

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

To run as a **desktop window** instead:

```bash
python desktop.py
```

## How to Use

1. **Select a backend** in the sidebar — Ollama (recommended) or mlx-lm
2. **Upload an image** — DICOM (`.dcm`), PNG, or JPEG
3. **Add context** (optional) — Describe symptoms or ask a specific question
4. **Click "Run Analysis"**
5. **Acknowledge the disclaimer** to view the AI's provisional findings

## Supported Image Formats

| Format | Details |
|--------|---------|
| **DICOM** (`.dcm`) | Full metadata extraction, HU calibration for CT, windowing presets, photometric inversion |
| **PNG** | Standard RGB or grayscale |
| **JPEG** | Standard RGB or grayscale |

## Project Structure

```
├── app.py              # Streamlit frontend and main entry point
├── preprocessing.py    # DICOM parsing, HU calibration, windowing
├── inference.py        # Ollama and mlx-lm backend integration
├── desktop.py          # Native desktop window launcher
├── requirements.txt    # Python dependencies
├── tests/              # Test suite (78 tests)
└── README.md
```

## Requirements

| Dependency | Purpose |
|------------|---------|
| [Streamlit](https://streamlit.io/) | Web UI |
| [pydicom](https://pydicom.github.io/) | DICOM file parsing |
| [Pillow](https://python-pillow.org/) | Image processing |
| [NumPy](https://numpy.org/) | Numerical operations |
| [pywebview](https://pywebview.flowrl.com/) | Desktop window mode |

## Running Tests

```bash
pip install pytest pytest-cov hypothesis

pytest tests/ -v
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
