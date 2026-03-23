"""
Medical Image Preprocessing Module

Handles DICOM parsing, HU calibration, windowing, and photometric inversion
for CT, X-ray, and MRI data before model inference.
"""

import numpy as np
import pydicom
from PIL import Image


def load_dicom(file_path_or_buffer):
    """Load a DICOM file from a path or file-like buffer."""
    ds = pydicom.dcmread(file_path_or_buffer)
    return ds


def extract_metadata(ds):
    """Extract relevant DICOM metadata for display and processing."""
    metadata = {}
    tag_map = {
        "PatientID": "PatientID",
        "PatientName": "PatientName",
        "Modality": "Modality",
        "StudyDescription": "StudyDescription",
        "SeriesDescription": "SeriesDescription",
        "BodyPartExamined": "BodyPartExamined",
        "PhotometricInterpretation": "PhotometricInterpretation",
        "RescaleSlope": "RescaleSlope",
        "RescaleIntercept": "RescaleIntercept",
        "WindowCenter": "WindowCenter",
        "WindowWidth": "WindowWidth",
        "BitsStored": "BitsStored",
        "Rows": "Rows",
        "Columns": "Columns",
    }
    for key, tag in tag_map.items():
        val = getattr(ds, tag, None)
        if val is not None:
            metadata[key] = str(val)
    return metadata


def apply_hu_calibration(pixel_array, ds):
    """Apply Hounsfield Unit calibration for CT images.

    HU = pixel_value * RescaleSlope + RescaleIntercept

    Expects pixel_array to already be a float array (the caller is responsible
    for the conversion to avoid redundant work).
    """
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    hu_image = pixel_array * float(slope) + float(intercept)
    return hu_image


def apply_windowing(image_array, window_center, window_width):
    """Apply windowing to map high-bit-depth values to 8-bit display range.

    Maps the range [center - width/2, center + width/2] to [0, 255].
    """
    wc = float(window_center)
    ww = float(window_width)
    if np.isclose(ww, 0.0):
        # For a zero-width window, create a binary image based on the center.
        return np.where(image_array > wc, 255, 0).astype(np.uint8)
    lower = wc - ww / 2.0
    upper = wc + ww / 2.0
    windowed = np.clip(image_array, lower, upper)
    windowed = ((windowed - lower) / ww * 255.0).astype(np.uint8)
    return windowed


def apply_photometric_inversion(image_array, ds):
    """Invert MONOCHROME1 images so bone=white and air=black (MONOCHROME2 convention)."""
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        if image_array.dtype == np.uint8:
            return 255 - image_array
        return image_array.max() - image_array
    return image_array


def _parse_window_value(val):
    """Parse a DICOM window value which may be a multi-value or single value."""
    if val is None:
        return None
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return float(val[0])
    return float(val)


def preprocess_dicom(ds):
    """Full preprocessing pipeline for a DICOM dataset.

    Returns an 8-bit numpy array ready for display/model input and the metadata dict.
    """
    pixel_array = ds.pixel_array.astype(np.float64)
    metadata = extract_metadata(ds)
    modality = getattr(ds, "Modality", "").upper()

    # Step 1: HU Calibration (primarily for CT)
    if modality == "CT" or hasattr(ds, "RescaleSlope"):
        pixel_array = apply_hu_calibration(pixel_array, ds)

    # Step 2: Windowing
    wc = _parse_window_value(getattr(ds, "WindowCenter", None))
    ww = _parse_window_value(getattr(ds, "WindowWidth", None))
    if wc is not None and ww is not None:
        image_8bit = apply_windowing(pixel_array, wc, ww)
    else:
        # Fallback: normalize to 0-255
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            image_8bit = ((pixel_array - pmin) / (pmax - pmin) * 255.0).astype(np.uint8)
        else:
            image_8bit = np.zeros_like(pixel_array, dtype=np.uint8)

    # Step 3: Photometric inversion
    image_8bit = apply_photometric_inversion(image_8bit, ds)

    return image_8bit, metadata


def preprocess_standard_image(uploaded_file):
    """Preprocess a standard image file (PNG, JPG, etc.) for model input."""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image), {}


def dicom_to_pil(image_8bit):
    """Convert an 8-bit numpy array to a PIL Image for display."""
    if len(image_8bit.shape) == 2:
        return Image.fromarray(image_8bit, mode="L")
    return Image.fromarray(image_8bit)
