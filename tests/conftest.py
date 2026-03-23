"""Shared test fixtures for the medical imaging analysis app."""

import io

import numpy as np
import pydicom
import pytest
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian


@pytest.fixture
def synthetic_ct_dicom():
    """Create a synthetic CT DICOM dataset with standard tags."""
    ds = Dataset()
    ds.PatientID = "TEST001"
    ds.PatientName = "Test^Patient"
    ds.Modality = "CT"
    ds.StudyDescription = "CT Abdomen"
    ds.SeriesDescription = "Axial 5mm"
    ds.BodyPartExamined = "ABDOMEN"
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    ds.PhotometricInterpretation = "MONOCHROME2"
    pixel_data = np.random.randint(-1024, 3000, (64, 64), dtype=np.int16)
    ds.PixelData = pixel_data.tobytes()
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


@pytest.fixture
def synthetic_xray_mono1():
    """Create a synthetic X-ray DICOM with MONOCHROME1 photometric interpretation."""
    ds = Dataset()
    ds.PatientID = "TEST002"
    ds.Modality = "CR"
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsStored = 12
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 0
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096
    ds.PhotometricInterpretation = "MONOCHROME1"
    pixel_data = np.random.randint(0, 4095, (64, 64), dtype=np.uint16)
    ds.PixelData = pixel_data.tobytes()
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


@pytest.fixture
def minimal_dicom():
    """Create a minimal DICOM with only pixel data, no optional tags."""
    ds = Dataset()
    ds.Rows = 32
    ds.Columns = 32
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    pixel_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    ds.PixelData = pixel_data.tobytes()
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


@pytest.fixture
def sample_png_buffer():
    """Create a synthetic RGB PNG in a BytesIO buffer."""
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def sample_grayscale_png_buffer():
    """Create a synthetic grayscale PNG in a BytesIO buffer."""
    arr = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def sample_jpeg_buffer():
    """Create a synthetic JPEG in a BytesIO buffer."""
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def mock_ollama_tags_response():
    """Standard Ollama /api/tags response with llama3.2-vision model."""
    return {
        "models": [
            {"name": "llama3.2-vision:latest", "size": 7365960935},
            {"name": "llava:latest", "size": 4733363377},
        ]
    }


@pytest.fixture
def mock_ollama_generate_response():
    """Standard Ollama /api/generate response."""
    return {
        "model": "llama3.2-vision",
        "response": (
            "**Step 1 – Modality Identification:**\n"
            "This is a PA Chest X-ray.\n\n"
            "**Step 2 – Observations:**\n"
            "No acute findings. Heart size is normal.\n\n"
            "**Step 3 – Diagnostic Synthesis:**\n"
            "Normal chest radiograph. Confidence: High.\n\n"
            "**Step 4 – Referral Recommendation:**\n"
            "General Practitioner for routine follow-up."
        ),
        "done": True,
    }
