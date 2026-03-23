# Comprehensive Test Plan — Medical Imaging Analysis App

## 1. Testing Toolchain

All tools installed locally and verified on Python 3.11.

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| **pytest** | 9.0.2 | Test runner and framework | `pip install pytest` |
| **pytest-cov** | 7.1.0 | Code coverage measurement | `pip install pytest-cov` |
| **pytest-mock** | 3.15.1 | Mocking/patching helpers | `pip install pytest-mock` |
| **pytest-xdist** | 3.8.0 | Parallel test execution | `pip install pytest-xdist` |
| **pytest-benchmark** | 5.2.3 | Performance benchmarking | `pip install pytest-benchmark` |
| **hypothesis** | 6.151.9 | Property-based / fuzz testing | `pip install hypothesis` |
| **responses** | 0.26.0 | Mock HTTP (Ollama API) | `pip install responses` |
| **ruff** | 0.15.7 | Fast linter + formatter | `pip install ruff` |
| **mypy** | 1.19.1 | Static type checking | `pip install mypy` |
| **bandit** | 1.9.4 | Security vulnerability scanner | `pip install bandit` |
| **streamlit** | 1.55.0 | Includes `AppTest` for UI testing | (app dependency) |

### Run Commands

```bash
# All unit/integration tests with coverage
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# Parallel execution
pytest tests/ -n auto

# Performance benchmarks only
pytest tests/test_performance.py --benchmark-only

# Linting
ruff check .

# Type checking
mypy --ignore-missing-imports app.py preprocessing.py inference.py

# Security scan
bandit -r . -x ./.git,./tests
```

---

## 2. Static Analysis Findings (Baseline)

Ran all three tools against the current codebase:

### Ruff — 5 issues (all auto-fixable)
| File | Issue | Severity |
|------|-------|----------|
| `app.py` | `import io` unused | Low |
| `app.py` | `import numpy as np` unused | Low |
| `app.py` | `SYSTEM_PROMPT` imported but unused | Low |
| `app.py` | `extract_metadata` imported but unused | Low |
| `inference.py` | `import json` unused | Low |

### Mypy — 26 type annotation gaps
- All 3 files lack function type annotations (`no-untyped-def`)
- No runtime type errors detected
- `requests` stubs need `types-requests` package

### Bandit — 0 security issues
- Clean scan across 447 lines of code

---

## 3. Functional Test Plan

### 3.1 Preprocessing Module (`tests/test_preprocessing.py`)

#### 3.1.1 DICOM Loading
| ID | Test Case | Input | Expected | Priority |
|----|-----------|-------|----------|----------|
| P-01 | Load valid DICOM file | Synthetic CT DICOM | Returns pydicom Dataset | High |
| P-02 | Load from file-like buffer | BytesIO with DICOM data | Returns pydicom Dataset | High |
| P-03 | Load corrupted DICOM | Random bytes | Raises `InvalidDicomError` | High |
| P-04 | Load empty file | 0-byte buffer | Raises exception | Medium |
| P-05 | Load non-DICOM file | PNG as .dcm | Raises `InvalidDicomError` | Medium |

#### 3.1.2 HU Calibration
| ID | Test Case | Input | Expected | Priority |
|----|-----------|-------|----------|----------|
| H-01 | Standard CT calibration | slope=1, intercept=-1024 | `HU = pixel * 1 + (-1024)` | High |
| H-02 | Non-unit slope | slope=2, intercept=0 | All values doubled | High |
| H-03 | Zero slope | slope=0, intercept=100 | All values = 100 | Medium |
| H-04 | Missing RescaleSlope | No attribute | Defaults to slope=1.0 | High |
| H-05 | Missing RescaleIntercept | No attribute | Defaults to intercept=0.0 | High |
| H-06 | Negative pixel values | Signed int16 input | Correct HU transform | Medium |
| H-07 | Large pixel values | 16-bit max (65535) | No overflow in float64 | Medium |

#### 3.1.3 Windowing
| ID | Test Case | Input | Expected | Priority |
|----|-----------|-------|----------|----------|
| W-01 | Standard bone window | WC=500, WW=2000 | Maps [-500,1500] → [0,255] | High |
| W-02 | Soft tissue window | WC=40, WW=400 | Maps [-160,240] → [0,255] | High |
| W-03 | Lung window | WC=-600, WW=1600 | Maps [-1400,200] → [0,255] | High |
| W-04 | Values below window | All pixels < lower bound | All output = 0 | High |
| W-05 | Values above window | All pixels > upper bound | All output = 255 | High |
| W-06 | Zero window width | WW=0 | Should not crash (edge case) | High |
| W-07 | Negative window center | WC=-500, WW=100 | Correct mapping | Medium |
| W-08 | Single-value image | All pixels = WC | All output = 127 or 128 | Medium |
| W-09 | Fallback normalization | No WC/WW tags | Normalizes min→0, max→255 | High |
| W-10 | Constant image fallback | All pixels same, no tags | Returns all-zero array | Medium |

#### 3.1.4 Photometric Inversion
| ID | Test Case | Input | Expected | Priority |
|----|-----------|-------|----------|----------|
| I-01 | MONOCHROME1 uint8 | 8-bit array, MONOCHROME1 tag | `255 - pixel` | High |
| I-02 | MONOCHROME2 passthrough | 8-bit array, MONOCHROME2 tag | Unchanged | High |
| I-03 | MONOCHROME1 non-uint8 | float64 array, MONOCHROME1 | `max - pixel` | Medium |
| I-04 | Missing PhotometricInterpretation | No tag | Defaults to MONOCHROME2 (no inversion) | High |
| I-05 | RGB image | 3-channel array | No inversion applied | Low |

#### 3.1.5 Full Pipeline (`preprocess_dicom`)
| ID | Test Case | Input | Expected | Priority |
|----|-----------|-------|----------|----------|
| FP-01 | CT with all tags | CT DICOM with slope/intercept/WC/WW | Correctly calibrated 8-bit output | High |
| FP-02 | X-ray MONOCHROME1 | CR modality, MONOCHROME1 | Inverted 8-bit output | High |
| FP-03 | MRI without rescale | MR modality, no slope | Windowing-only output | High |
| FP-04 | Minimal DICOM | Only pixel data, no optional tags | Fallback normalization | High |
| FP-05 | 16-bit depth image | BitsStored=16 | Correctly mapped to 8-bit | Medium |
| FP-06 | Multi-value window tags | WC=[40,500], WW=[400,2000] | Uses first value | Medium |

#### 3.1.6 Standard Image Processing
| ID | Test Case | Input | Expected | Priority |
|----|-----------|-------|----------|----------|
| SI-01 | Load PNG | Valid PNG file | RGB numpy array, empty metadata | High |
| SI-02 | Load JPEG | Valid JPEG file | RGB numpy array, empty metadata | High |
| SI-03 | Grayscale PNG | Single-channel PNG | Converted to RGB | Medium |
| SI-04 | RGBA image | 4-channel PNG | Converted to RGB | Medium |

#### 3.1.7 Metadata Extraction
| ID | Test Case | Input | Expected | Priority |
|----|-----------|-------|----------|----------|
| M-01 | Full metadata | DICOM with all 13 tags | Dict with all 13 keys | High |
| M-02 | Partial metadata | DICOM with only Modality | Dict with only Modality | High |
| M-03 | Empty dataset | Dataset with no tags | Empty dict | Medium |

#### 3.1.8 Property-Based Tests (Hypothesis)
| ID | Property | Strategy | Priority |
|----|----------|----------|----------|
| PB-01 | Windowing always returns uint8 in [0,255] | Random arrays, random WC/WW | High |
| PB-02 | HU calibration preserves array shape | Random arrays, random slopes | High |
| PB-03 | Photometric inversion is self-inverse | Random uint8 arrays | High |
| PB-04 | dicom_to_pil handles any 2D uint8 array | Random 2D arrays | Medium |

---

### 3.2 Inference Module (`tests/test_inference.py`)

#### 3.2.1 Image Encoding
| ID | Test Case | Input | Expected | Priority |
|----|-----------|-------|----------|----------|
| IE-01 | Encode numpy RGB array | 100x100x3 uint8 | Valid base64 PNG | High |
| IE-02 | Encode numpy grayscale | 100x100 uint8 | Valid base64 PNG | High |
| IE-03 | Encode PIL Image | PIL RGB Image | Valid base64 PNG | High |
| IE-04 | Encode RGBA image | RGBA mode PIL | Converted to RGB, valid b64 | Medium |
| IE-05 | Roundtrip decode | Encode then decode | Identical dimensions | High |

#### 3.2.2 OllamaBackend
| ID | Test Case | Mock | Expected | Priority |
|----|-----------|------|----------|----------|
| OB-01 | is_available — server up, model present | 200 with model list | Returns True | High |
| OB-02 | is_available — server up, model absent | 200 without model | Returns False | High |
| OB-03 | is_available — server down | ConnectionError | Returns False | High |
| OB-04 | list_models — returns models | 200 with 3 models | List of 3 names | High |
| OB-05 | list_models — server down | ConnectionError | Returns [] | Medium |
| OB-06 | analyze — successful response | 200 with response text | Returns response string | High |
| OB-07 | analyze — with metadata | 200 | Metadata appended to prompt | High |
| OB-08 | analyze — HTTP error | 500 status | Raises HTTPError | High |
| OB-09 | analyze — timeout | Timeout | Raises Timeout exception | Medium |
| OB-10 | analyze — empty response | 200, no "response" key | Returns "No response received." | Medium |
| OB-11 | Verify system prompt sent | Any | Payload contains SYSTEM_PROMPT | High |
| OB-12 | Verify image sent as base64 | Any | Payload "images" has 1 entry | High |

#### 3.2.3 MlxLmBackend
| ID | Test Case | Mock | Expected | Priority |
|----|-----------|------|----------|----------|
| ML-01 | is_available — mlx_lm installed | Mock import success | Returns True | High |
| ML-02 | is_available — mlx_lm missing | Mock ImportError | Returns False | High |
| ML-03 | analyze — successful | Mock generate() | Returns generated text | High |
| ML-04 | analyze — numpy input | Mock generate() | Converts to PIL first | Medium |
| ML-05 | Lazy model loading | Two analyze calls | load() called once | High |

#### 3.2.4 Backend Selection
| ID | Test Case | Condition | Expected | Priority |
|----|-----------|-----------|----------|----------|
| BS-01 | Both available | Ollama and mlx-lm up | Returns OllamaBackend | High |
| BS-02 | Only Ollama | mlx-lm not installed | Returns OllamaBackend | High |
| BS-03 | Only mlx-lm | Ollama down | Returns MlxLmBackend | High |
| BS-04 | Neither available | Both down | Returns None | High |

#### 3.2.5 System Prompt Validation
| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| SP-01 | Prompt contains Step 1 (Modality) | "Modality Identification" present | High |
| SP-02 | Prompt contains Step 2 (Observation) | "Observations" present | High |
| SP-03 | Prompt contains Step 3 (Synthesis) | "Diagnostic Synthesis" present | High |
| SP-04 | Prompt contains Step 4 (Referral) | "Referral Recommendation" present | High |
| SP-05 | Prompt contains disclaimer | "PROVISIONAL" present | High |
| SP-06 | Referral mentions Internist | "Internist" present | Medium |
| SP-07 | Referral mentions Surgeon | "Surgeon" present | Medium |
| SP-08 | Referral mentions GP | "General Practitioner" present | Medium |

---

### 3.3 Streamlit App (`tests/test_app.py`)

Using Streamlit's built-in `AppTest` framework (`from streamlit.testing.v1 import AppTest`).

#### 3.3.1 Page Structure
| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| UI-01 | App loads without error | No exceptions on startup | High |
| UI-02 | Title renders | "Medical Imaging Analysis" in title | High |
| UI-03 | Sidebar renders | Backend config radio present | High |
| UI-04 | File uploader renders | Upload widget present | High |
| UI-05 | Info message when no file | "Upload a medical image" shown | Medium |
| UI-06 | Supported formats expander | Lists DICOM, PNG, JPEG | Low |

#### 3.3.2 Session State
| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| SS-01 | Initial state | analysis_result=None, acknowledged=False | High |
| SS-02 | State persists across reruns | Values retained | Medium |

#### 3.3.3 Governor Pattern / Disclaimer
| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| GP-01 | Disclaimer shown before acknowledgment | Warning with disclaimer text visible | High |
| GP-02 | Results hidden before acknowledgment | Technical findings NOT shown | High |
| GP-03 | Acknowledgment button exists | "I understand" button present | High |
| GP-04 | After acknowledgment, results shown | Technical findings visible | High |
| GP-05 | Disclaimer text contains "PROVISIONAL" | Exact text match | High |
| GP-06 | Disclaimer text contains "NOT A MEDICAL DIAGNOSIS" | Exact text match | High |

#### 3.3.4 Patient-Friendly Output
| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| PF-01 | Summary generated from results | Contains "What This Means for You" | High |
| PF-02 | Summary contains safety reminders | "preliminary", "qualified doctor" | High |
| PF-03 | Summary includes original findings | Technical text embedded | Medium |

---

## 4. Non-Functional Test Plan

### 4.1 Performance (`tests/test_performance.py`)

Using `pytest-benchmark` and manual timing.

| ID | Test Case | Metric | Target | Priority |
|----|-----------|--------|--------|----------|
| PF-01 | HU calibration 512x512 | Wall time | < 10ms | High |
| PF-02 | Windowing 512x512 | Wall time | < 10ms | High |
| PF-03 | Full DICOM preprocessing 512x512 | Wall time | < 50ms | High |
| PF-04 | Base64 encoding 512x512 RGB | Wall time | < 50ms | Medium |
| PF-05 | Metadata extraction | Wall time | < 5ms | Low |
| PF-06 | Standard image loading 1024x1024 | Wall time | < 100ms | Medium |
| PF-07 | HU calibration 2048x2048 (large CT) | Wall time | < 200ms | Medium |
| PF-08 | Windowing 2048x2048 | Wall time | < 200ms | Medium |

### 4.2 Memory Usage

| ID | Test Case | Metric | Target | Priority |
|----|-----------|--------|--------|----------|
| ME-01 | Preprocess 512x512 DICOM | Peak memory delta | < 50MB | High |
| ME-02 | Preprocess 2048x2048 DICOM | Peak memory delta | < 500MB | Medium |
| ME-03 | Base64 encode large image | Peak memory delta | < 100MB | Medium |
| ME-04 | No memory leak on repeated preprocessing | Memory after 100 calls | Stable | High |

### 4.3 Reliability / Error Handling

| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| RE-01 | Corrupted DICOM does not crash app | Error message displayed | High |
| RE-02 | Ollama unavailable does not crash app | "not available" error shown | High |
| RE-03 | Ollama timeout handled gracefully | Exception caught, error shown | High |
| RE-04 | Invalid image format uploaded | Error message, not crash | High |
| RE-05 | Zero-byte file uploaded | Graceful error handling | Medium |
| RE-06 | Very large file (100MB+) | Handles or rejects gracefully | Medium |

### 4.4 Security

| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| SE-01 | Bandit scan passes | 0 high/medium issues | High |
| SE-02 | No external network calls in preprocessing | Verified by code review | High |
| SE-03 | Ollama requests go only to configured URL | No other outbound requests | High |
| SE-04 | Patient metadata not logged to console | No prints/logs with PII | Medium |
| SE-05 | File type validated beyond extension | Content-based validation | Medium |

### 4.5 Code Quality

| ID | Test Case | Metric | Target | Priority |
|----|-----------|--------|--------|----------|
| CQ-01 | Ruff lint passes | 0 errors | High |
| CQ-02 | Mypy passes (ignore-missing-imports) | 0 errors (excl. missing stubs) | Medium |
| CQ-03 | Test coverage | Line coverage | ≥ 85% | High |
| CQ-04 | Branch coverage | Branch coverage | ≥ 75% | Medium |

### 4.6 Compatibility

| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| CO-01 | Python 3.10 compatibility | All tests pass | Medium |
| CO-02 | Python 3.11 compatibility | All tests pass | High |
| CO-03 | Python 3.12 compatibility | All tests pass | Medium |
| CO-04 | Runs without mlx-lm installed | Ollama-only mode works | High |
| CO-05 | Runs without Ollama running | mlx-lm-only mode works | Medium |
| CO-06 | Runs without either backend | Graceful error messaging | High |

### 4.7 Privacy

| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| PR-01 | No outbound network calls during preprocessing | Verified via mocking | High |
| PR-02 | Patient identifiers hidden by default | Checkbox required to show | High |
| PR-03 | Inference stays local (Ollama localhost) | Default URL is localhost | High |
| PR-04 | No telemetry or analytics | No tracking code present | Medium |

---

## 5. Test Data Strategy

### 5.1 Synthetic DICOM Generation

Use pydicom to create synthetic DICOM files programmatically in test fixtures:

```python
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import numpy as np
import tempfile

def create_synthetic_ct_dicom(rows=512, cols=512):
    """Create a synthetic CT DICOM for testing."""
    ds = Dataset()
    ds.PatientID = "TEST001"
    ds.PatientName = "Test^Patient"
    ds.Modality = "CT"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 1  # signed
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.random.randint(-1024, 3000, (rows, cols), dtype=np.int16).tobytes()
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds
```

### 5.2 Test Image Generation

```python
from PIL import Image
import io

def create_test_png(width=256, height=256):
    """Create a synthetic PNG for testing."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf
```

---

## 6. Test Directory Structure

```
tests/
├── conftest.py               # Shared fixtures (synthetic DICOMs, mock backends)
├── test_preprocessing.py     # P-*, H-*, W-*, I-*, FP-*, SI-*, M-* tests
├── test_inference.py         # IE-*, OB-*, ML-*, BS-*, SP-* tests
├── test_app.py               # UI-*, SS-*, GP-*, PF-* tests (Streamlit AppTest)
├── test_performance.py       # PF-*, ME-* benchmarks
├── test_property.py          # PB-* hypothesis-based tests
└── test_integration.py       # End-to-end with mocked backend
```

---

## 7. Execution Priority

### Phase 1 — Critical (Run First)
- All **High** priority functional tests (P-01..03, H-01..05, W-01..06, W-09, I-01..02, I-04, FP-01..04, OB-01..03, OB-06, OB-08)
- System prompt validation (SP-01..05)
- Governor pattern tests (GP-01..06)
- Ruff lint fix (CQ-01)
- Bandit security scan (SE-01)

### Phase 2 — Important (Run Second)
- Remaining **High** tests (IE-*, BS-*, ML-*, SI-01..02, UI-01..04, SS-01)
- Performance benchmarks (PF-01..03)
- Memory tests (ME-01, ME-04)
- Error handling (RE-01..04)
- Coverage target ≥ 85% (CQ-03)

### Phase 3 — Comprehensive (Run Last)
- All **Medium** and **Low** priority tests
- Property-based tests (PB-01..04)
- Compatibility tests (CO-01..06)
- Privacy validation (PR-01..04)
- Large image performance (PF-07..08)

---

## 8. Pre-Existing Issues to Fix Before Testing

Found during static analysis — should be fixed to ensure clean test runs:

1. **Unused imports in `app.py`**: `io`, `numpy`, `SYSTEM_PROMPT`, `extract_metadata`
2. **Unused import in `inference.py`**: `json`
3. **`opencv-python-headless` in requirements.txt**: Never imported in any file
4. **Zero window width bug**: `apply_windowing()` with `ww=0` causes division by zero
5. **Missing error handling in `app.py`**: `load_dicom()` call has no try/except
6. **Double conversion in `preprocess_dicom`**: pixel_array converted to float64 on line 94, then again in `apply_hu_calibration`
