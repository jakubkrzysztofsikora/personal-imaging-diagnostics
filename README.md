# Medical Imaging Analysis

A local, privacy-preserving application for analyzing medical images (X-rays, CTs, MRIs) using a Vision-Language Model. Built with Streamlit and powered by Ollama or mlx-lm for fully local inference.

> **Disclaimer:** This tool is for educational and informational purposes only. It does not provide medical diagnoses. Always consult a qualified healthcare professional.

## Features

- **DICOM Support** — Full parsing via pydicom with automatic HU calibration, windowing, and photometric inversion
- **Standard Image Support** — PNG and JPEG uploads for non-DICOM workflows
- **4-Step AI Analysis** — Structured Chain-of-Thought: Modality ID → Observations → Diagnostic Synthesis → Referral
- **Governor Pattern** — All AI findings are provisional and gated behind an acknowledgment step
- **Privacy First** — All processing runs locally; no external API calls

## Prerequisites

- Python 3.10+
- **One of the following backends:**
  - [Ollama](https://ollama.com/) with a vision model (recommended)
  - [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon

### Setting Up Ollama (Recommended)

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Pull a vision-capable model:
   ```bash
   ollama pull llama3.2-vision
   ```
3. Ensure Ollama is running (`ollama serve` or the desktop app)

### Setting Up mlx-lm (Apple Silicon Only)

```bash
pip install mlx-lm
```

The app will download the model on first use (default: `mlx-community/GLM-4.5V-9B-4bit`).

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd personal-imaging-diagnostics

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Select your inference backend in the sidebar (Ollama or mlx-lm)
2. Upload a medical image (DICOM, PNG, or JPEG)
3. Optionally add context or a specific question
4. Click **Run Analysis**
5. Acknowledge the disclaimer to view provisional findings

## Project Structure

```
├── app.py              # Streamlit frontend and main application
├── preprocessing.py    # DICOM parsing, HU calibration, windowing, inversion
├── inference.py        # Ollama and mlx-lm backend integration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Preprocessing Pipeline

For DICOM files, the app applies the following pipeline automatically:

1. **HU Calibration** (CT scans): `HU = pixel_value × RescaleSlope + RescaleIntercept`
2. **Windowing**: Maps the `[center - width/2, center + width/2]` range to 8-bit `[0, 255]`
3. **Photometric Inversion**: Converts MONOCHROME1 → MONOCHROME2 so bone appears white and air appears black
