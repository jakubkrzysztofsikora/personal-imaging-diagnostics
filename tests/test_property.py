"""Property-based tests using Hypothesis for fuzz testing."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pydicom.dataset import Dataset

from preprocessing import (
    apply_hu_calibration,
    apply_photometric_inversion,
    apply_windowing,
    dicom_to_pil,
)

# ---------------------------------------------------------------------------
# PB-01: Windowing always returns uint8 in [0, 255]
# ---------------------------------------------------------------------------

@given(
    arr=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=128),
            st.integers(min_value=1, max_value=128),
        ),
        elements=st.floats(min_value=-5000, max_value=5000, allow_nan=False, allow_infinity=False),
    ),
    wc=st.floats(min_value=-2000, max_value=2000, allow_nan=False, allow_infinity=False),
    ww=st.floats(min_value=1, max_value=5000, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_windowing_always_returns_uint8(arr, wc, ww):
    """PB-01: Windowing output is always uint8 with values in [0, 255]."""
    result = apply_windowing(arr, wc, ww)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
    assert result.shape == arr.shape


# ---------------------------------------------------------------------------
# PB-02: HU calibration preserves array shape
# ---------------------------------------------------------------------------

@given(
    arr=arrays(
        dtype=np.int16,
        shape=st.tuples(
            st.integers(min_value=1, max_value=128),
            st.integers(min_value=1, max_value=128),
        ),
    ),
    slope=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    intercept=st.floats(min_value=-5000, max_value=5000, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_hu_calibration_preserves_shape(arr, slope, intercept):
    """PB-02: HU calibration preserves array shape."""
    ds = Dataset()
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    result = apply_hu_calibration(arr, ds)
    assert result.shape == arr.shape


# ---------------------------------------------------------------------------
# PB-03: Photometric inversion is self-inverse (for uint8)
# ---------------------------------------------------------------------------

@given(
    arr=arrays(
        dtype=np.uint8,
        shape=st.tuples(
            st.integers(min_value=1, max_value=64),
            st.integers(min_value=1, max_value=64),
        ),
    ),
)
@settings(max_examples=200)
def test_photometric_inversion_is_self_inverse(arr):
    """PB-03: Applying MONOCHROME1 inversion twice returns original."""
    ds = Dataset()
    ds.PhotometricInterpretation = "MONOCHROME1"
    inverted = apply_photometric_inversion(arr, ds)
    double_inverted = apply_photometric_inversion(inverted, ds)
    np.testing.assert_array_equal(double_inverted, arr)


# ---------------------------------------------------------------------------
# PB-04: dicom_to_pil handles any 2D uint8 array
# ---------------------------------------------------------------------------

@given(
    arr=arrays(
        dtype=np.uint8,
        shape=st.tuples(
            st.integers(min_value=1, max_value=256),
            st.integers(min_value=1, max_value=256),
        ),
    ),
)
@settings(max_examples=100)
def test_dicom_to_pil_handles_2d(arr):
    """PB-04: dicom_to_pil converts any 2D uint8 array to PIL Image."""
    img = dicom_to_pil(arr)
    assert img.mode == "L"
    assert img.size == (arr.shape[1], arr.shape[0])
