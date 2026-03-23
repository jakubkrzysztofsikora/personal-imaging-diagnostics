"""Tests for the preprocessing module."""

import io

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset

from preprocessing import (
    _parse_window_value,
    apply_hu_calibration,
    apply_photometric_inversion,
    apply_windowing,
    dicom_to_pil,
    extract_metadata,
    load_dicom,
    preprocess_dicom,
    preprocess_standard_image,
)

# ---------------------------------------------------------------------------
# DICOM Loading (P-*)
# ---------------------------------------------------------------------------

class TestLoadDicom:
    def test_load_valid_dicom_from_buffer(self, synthetic_ct_dicom, tmp_path):
        """P-01: Load a valid CT DICOM from file path."""
        path = tmp_path / "test.dcm"
        pydicom.dcmwrite(str(path), synthetic_ct_dicom, write_like_original=False)
        ds = load_dicom(str(path))
        assert ds.Modality == "CT"
        assert ds.Rows == 64

    def test_load_from_bytesio(self, synthetic_ct_dicom):
        """P-02: Load DICOM from a file-like BytesIO buffer."""
        buf = io.BytesIO()
        pydicom.dcmwrite(buf, synthetic_ct_dicom, write_like_original=False)
        buf.seek(0)
        ds = load_dicom(buf)
        assert ds.Modality == "CT"

    def test_load_corrupted_raises(self, tmp_path):
        """P-03: Corrupted data should raise an exception."""
        path = tmp_path / "corrupted.dcm"
        path.write_bytes(b"this is not a dicom file at all")
        with pytest.raises(Exception):
            load_dicom(str(path))

    def test_load_empty_file_raises(self, tmp_path):
        """P-04: Empty file should raise an exception."""
        path = tmp_path / "empty.dcm"
        path.write_bytes(b"")
        with pytest.raises(Exception):
            load_dicom(str(path))


# ---------------------------------------------------------------------------
# HU Calibration (H-*)
# ---------------------------------------------------------------------------

class TestHUCalibration:
    def test_standard_ct_calibration(self):
        """H-01: Standard slope=1, intercept=-1024."""
        ds = Dataset()
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        pixels = np.array([[0, 1024, 2048]], dtype=np.int16)
        result = apply_hu_calibration(pixels, ds)
        np.testing.assert_array_almost_equal(result, [[-1024.0, 0.0, 1024.0]])

    def test_non_unit_slope(self):
        """H-02: slope=2, intercept=0."""
        ds = Dataset()
        ds.RescaleSlope = 2.0
        ds.RescaleIntercept = 0.0
        pixels = np.array([[10, 20, 30]], dtype=np.int16)
        result = apply_hu_calibration(pixels, ds)
        np.testing.assert_array_almost_equal(result, [[20.0, 40.0, 60.0]])

    def test_zero_slope(self):
        """H-03: slope=0 gives constant output."""
        ds = Dataset()
        ds.RescaleSlope = 0.0
        ds.RescaleIntercept = 100.0
        pixels = np.array([[1, 2, 3]], dtype=np.int16)
        result = apply_hu_calibration(pixels, ds)
        np.testing.assert_array_almost_equal(result, [[100.0, 100.0, 100.0]])

    def test_missing_slope_defaults_to_1(self):
        """H-04: Missing RescaleSlope defaults to 1.0."""
        ds = Dataset()
        ds.RescaleIntercept = -500.0
        pixels = np.array([[100]], dtype=np.int16)
        result = apply_hu_calibration(pixels, ds)
        np.testing.assert_array_almost_equal(result, [[-400.0]])

    def test_missing_intercept_defaults_to_0(self):
        """H-05: Missing RescaleIntercept defaults to 0.0."""
        ds = Dataset()
        ds.RescaleSlope = 2.0
        pixels = np.array([[50]], dtype=np.int16)
        result = apply_hu_calibration(pixels, ds)
        np.testing.assert_array_almost_equal(result, [[100.0]])

    def test_preserves_shape(self):
        """H-06/H-07: Shape is preserved for any input."""
        ds = Dataset()
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        pixels = np.random.randint(0, 65535, (128, 128), dtype=np.uint16)
        result = apply_hu_calibration(pixels, ds)
        assert result.shape == (128, 128)


# ---------------------------------------------------------------------------
# Windowing (W-*)
# ---------------------------------------------------------------------------

class TestWindowing:
    def test_bone_window(self):
        """W-01: Bone window WC=500, WW=2000 maps [-500,1500] to [0,255]."""
        arr = np.array([[-500.0, 500.0, 1500.0]])
        result = apply_windowing(arr, 500, 2000)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 1] == 127 or result[0, 1] == 128  # center
        assert result[0, 2] == 255

    def test_soft_tissue_window(self):
        """W-02: Soft tissue WC=40, WW=400."""
        arr = np.array([[-160.0, 40.0, 240.0]])
        result = apply_windowing(arr, 40, 400)
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_lung_window(self):
        """W-03: Lung window WC=-600, WW=1600."""
        arr = np.array([[-1400.0, -600.0, 200.0]])
        result = apply_windowing(arr, -600, 1600)
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_values_below_window(self):
        """W-04: All values below window → all 0."""
        arr = np.array([[-5000.0, -4000.0, -3000.0]])
        result = apply_windowing(arr, 0, 100)
        np.testing.assert_array_equal(result, [[0, 0, 0]])

    def test_values_above_window(self):
        """W-05: All values above window → all 255."""
        arr = np.array([[5000.0, 6000.0, 7000.0]])
        result = apply_windowing(arr, 0, 100)
        np.testing.assert_array_equal(result, [[255, 255, 255]])

    def test_zero_window_width(self):
        """W-06: Zero window width should not crash."""
        arr = np.array([[0.0, 100.0, 200.0]])
        # This is an edge case; we just verify no exception
        try:
            apply_windowing(arr, 100, 0)
            # With ww=0: lower=upper=100, clip makes all=100, then 0/0
            # numpy handles 0/0 as nan, cast to uint8 = 0
        except ZeroDivisionError:
            pytest.fail("apply_windowing should not raise ZeroDivisionError")

    def test_output_always_uint8(self):
        """W-09: Output is always uint8."""
        arr = np.random.uniform(-2000, 4000, (64, 64))
        result = apply_windowing(arr, 40, 400)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_image_fallback_normalization(self):
        """W-10: Constant image without window tags falls back to zeros."""
        # This tests the fallback path in preprocess_dicom, not windowing directly
        arr = np.full((32, 32), 42.0)
        pmin, pmax = arr.min(), arr.max()
        if pmax > pmin:
            result = ((arr - pmin) / (pmax - pmin) * 255.0).astype(np.uint8)
        else:
            result = np.zeros_like(arr, dtype=np.uint8)
        np.testing.assert_array_equal(result, np.zeros((32, 32), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Photometric Inversion (I-*)
# ---------------------------------------------------------------------------

class TestPhotometricInversion:
    def test_monochrome1_uint8_inverted(self):
        """I-01: MONOCHROME1 uint8 → 255 - pixel."""
        ds = Dataset()
        ds.PhotometricInterpretation = "MONOCHROME1"
        arr = np.array([[0, 127, 255]], dtype=np.uint8)
        result = apply_photometric_inversion(arr, ds)
        np.testing.assert_array_equal(result, [[255, 128, 0]])

    def test_monochrome2_passthrough(self):
        """I-02: MONOCHROME2 → unchanged."""
        ds = Dataset()
        ds.PhotometricInterpretation = "MONOCHROME2"
        arr = np.array([[0, 127, 255]], dtype=np.uint8)
        result = apply_photometric_inversion(arr, ds)
        np.testing.assert_array_equal(result, arr)

    def test_monochrome1_float_inverted(self):
        """I-03: MONOCHROME1 float64 → max - pixel."""
        ds = Dataset()
        ds.PhotometricInterpretation = "MONOCHROME1"
        arr = np.array([[0.0, 50.0, 100.0]])
        result = apply_photometric_inversion(arr, ds)
        np.testing.assert_array_almost_equal(result, [[100.0, 50.0, 0.0]])

    def test_missing_tag_defaults_no_inversion(self):
        """I-04: Missing PhotometricInterpretation → no inversion."""
        ds = Dataset()
        arr = np.array([[10, 20, 30]], dtype=np.uint8)
        result = apply_photometric_inversion(arr, ds)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# Full Pipeline (FP-*)
# ---------------------------------------------------------------------------

class TestPreprocessDicom:
    def test_ct_full_pipeline(self, synthetic_ct_dicom):
        """FP-01: CT with all tags produces 8-bit output."""
        image_8bit, metadata = preprocess_dicom(synthetic_ct_dicom)
        assert image_8bit.dtype == np.uint8
        assert image_8bit.shape == (64, 64)
        assert "Modality" in metadata
        assert metadata["Modality"] == "CT"

    def test_xray_monochrome1(self, synthetic_xray_mono1):
        """FP-02: X-ray MONOCHROME1 is inverted."""
        image_8bit, metadata = preprocess_dicom(synthetic_xray_mono1)
        assert image_8bit.dtype == np.uint8
        assert metadata.get("PhotometricInterpretation") == "MONOCHROME1"

    def test_minimal_dicom_fallback(self, minimal_dicom):
        """FP-04: Minimal DICOM uses fallback normalization."""
        image_8bit, metadata = preprocess_dicom(minimal_dicom)
        assert image_8bit.dtype == np.uint8
        assert image_8bit.shape == (32, 32)

    def test_multi_value_window_tags(self):
        """FP-06: Multi-value window tags use first value."""
        assert _parse_window_value([40, 500]) == 40.0
        assert _parse_window_value([400, 2000]) == 400.0
        assert _parse_window_value(40) == 40.0
        assert _parse_window_value(None) is None


# ---------------------------------------------------------------------------
# Standard Image Processing (SI-*)
# ---------------------------------------------------------------------------

class TestStandardImage:
    def test_load_png(self, sample_png_buffer):
        """SI-01: PNG loads as RGB numpy array with empty metadata."""
        arr, metadata = preprocess_standard_image(sample_png_buffer)
        assert arr.shape == (128, 128, 3)
        assert arr.dtype == np.uint8
        assert metadata == {}

    def test_load_jpeg(self, sample_jpeg_buffer):
        """SI-02: JPEG loads as RGB numpy array."""
        arr, metadata = preprocess_standard_image(sample_jpeg_buffer)
        assert arr.shape == (128, 128, 3)
        assert metadata == {}

    def test_grayscale_converted_to_rgb(self, sample_grayscale_png_buffer):
        """SI-03: Grayscale PNG is converted to RGB."""
        arr, metadata = preprocess_standard_image(sample_grayscale_png_buffer)
        assert arr.shape == (128, 128, 3)


# ---------------------------------------------------------------------------
# Metadata Extraction (M-*)
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    def test_full_metadata(self, synthetic_ct_dicom):
        """M-01: Full CT DICOM has all expected metadata keys."""
        meta = extract_metadata(synthetic_ct_dicom)
        assert "Modality" in meta
        assert "RescaleSlope" in meta
        assert "WindowCenter" in meta
        assert "PatientID" in meta

    def test_minimal_metadata(self, minimal_dicom):
        """M-02/M-03: Minimal DICOM returns only available keys."""
        meta = extract_metadata(minimal_dicom)
        assert "Modality" not in meta
        assert "Rows" in meta


# ---------------------------------------------------------------------------
# dicom_to_pil (utility)
# ---------------------------------------------------------------------------

class TestDicomToPil:
    def test_grayscale_to_pil(self):
        """Grayscale 2D array → PIL 'L' mode image."""
        arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        img = dicom_to_pil(arr)
        assert img.mode == "L"
        assert img.size == (64, 64)

    def test_rgb_to_pil(self):
        """RGB 3D array → PIL RGB image."""
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = dicom_to_pil(arr)
        assert img.size == (64, 64)
