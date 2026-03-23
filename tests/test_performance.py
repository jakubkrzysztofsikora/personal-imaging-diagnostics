"""Performance benchmarks using pytest-benchmark."""

import numpy as np
from pydicom.dataset import Dataset

from inference import _image_to_base64
from preprocessing import (
    apply_hu_calibration,
    apply_windowing,
    extract_metadata,
    preprocess_dicom,
)


def _make_ct_dataset(rows, cols):
    """Helper to create a CT dataset of given dimensions."""
    ds = Dataset()
    ds.Modality = "CT"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.random.randint(-1024, 3000, (rows, cols), dtype=np.int16).tobytes()
    ds.file_meta = Dataset()
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


# ---------------------------------------------------------------------------
# PF-01: HU calibration 512x512
# ---------------------------------------------------------------------------

def test_hu_calibration_512(benchmark):
    ds = Dataset()
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    arr = np.random.randint(-1024, 3000, (512, 512), dtype=np.int16)
    benchmark(apply_hu_calibration, arr, ds)


# ---------------------------------------------------------------------------
# PF-02: Windowing 512x512
# ---------------------------------------------------------------------------

def test_windowing_512(benchmark):
    arr = np.random.uniform(-1024, 3000, (512, 512))
    benchmark(apply_windowing, arr, 40, 400)


# ---------------------------------------------------------------------------
# PF-03: Full DICOM preprocessing 512x512
# ---------------------------------------------------------------------------

def test_full_preprocess_512(benchmark):
    ds = _make_ct_dataset(512, 512)
    benchmark(preprocess_dicom, ds)


# ---------------------------------------------------------------------------
# PF-04: Base64 encoding 512x512 RGB
# ---------------------------------------------------------------------------

def test_base64_encode_512(benchmark):
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    benchmark(_image_to_base64, arr)


# ---------------------------------------------------------------------------
# PF-05: Metadata extraction
# ---------------------------------------------------------------------------

def test_metadata_extraction(benchmark):
    ds = _make_ct_dataset(64, 64)
    benchmark(extract_metadata, ds)


# ---------------------------------------------------------------------------
# PF-07: HU calibration 2048x2048
# ---------------------------------------------------------------------------

def test_hu_calibration_2048(benchmark):
    ds = Dataset()
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    arr = np.random.randint(-1024, 3000, (2048, 2048), dtype=np.int16)
    benchmark.pedantic(apply_hu_calibration, args=(arr, ds), rounds=5)


# ---------------------------------------------------------------------------
# PF-08: Windowing 2048x2048
# ---------------------------------------------------------------------------

def test_windowing_2048(benchmark):
    arr = np.random.uniform(-1024, 3000, (2048, 2048))
    benchmark.pedantic(apply_windowing, args=(arr, 40, 400), rounds=5)
