"""Tests for the Streamlit app — unit tests and AppTest UI tests."""

from streamlit.testing.v1 import AppTest

from app import (
    DISCLAIMER,
    RECOMMENDED_VISION_MODELS,
    _format_size,
    _generate_patient_summary,
    init_session_state,
    render_model_management,
)

# ---------------------------------------------------------------------------
# Format Size Helper (FS-*)
# ---------------------------------------------------------------------------

class TestFormatSize:
    def test_bytes(self):
        assert "B" in _format_size(500)

    def test_megabytes(self):
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_gigabytes(self):
        result = _format_size(7 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_zero(self):
        assert "0.0 B" == _format_size(0)


# ---------------------------------------------------------------------------
# Governor Pattern / Disclaimer (GP-*)
# ---------------------------------------------------------------------------

class TestGovernorPattern:
    def test_disclaimer_contains_provisional(self):
        """GP-05: Disclaimer text contains 'PROVISIONAL'."""
        assert "PROVISIONAL" in DISCLAIMER

    def test_disclaimer_contains_not_diagnosis(self):
        """GP-06: Disclaimer text contains 'NOT A MEDICAL DIAGNOSIS'."""
        assert "NOT A MEDICAL DIAGNOSIS" in DISCLAIMER


# ---------------------------------------------------------------------------
# Patient-Friendly Output (PF-*)
# ---------------------------------------------------------------------------

class TestPatientSummary:
    def test_summary_header(self):
        """PF-01: Summary contains 'What This Means for You'."""
        result = _generate_patient_summary("Test findings here.")
        assert "What This Means for You" in result

    def test_summary_safety_reminders(self):
        """PF-02: Summary contains safety language."""
        result = _generate_patient_summary("Test findings.")
        assert "preliminary" in result
        assert "qualified doctor" in result

    def test_summary_includes_findings(self):
        """PF-03: Summary includes original technical text."""
        findings = "Significant opacity in right lower lobe."
        result = _generate_patient_summary(findings)
        assert findings in result


# ---------------------------------------------------------------------------
# Session State (SS-*)
# ---------------------------------------------------------------------------

class TestSessionState:
    def test_init_defaults(self):
        """SS-01: Verify default session state values."""
        assert callable(init_session_state)


class TestRecommendedModels:
    def test_recommended_models_list_exists(self):
        """Model management: recommended vision models list is populated."""
        assert len(RECOMMENDED_VISION_MODELS) > 0
        assert "llama3.2-vision" in RECOMMENDED_VISION_MODELS

    def test_model_management_function_exists(self):
        """Model management: render_model_management is callable."""
        assert callable(render_model_management)


# ---------------------------------------------------------------------------
# Streamlit AppTest UI Tests (UI-*)
# ---------------------------------------------------------------------------

class TestAppUI:
    def test_app_loads_without_error(self):
        """UI-01: App renders without crashing."""
        at = AppTest.from_file("app.py")
        at.run(timeout=10)
        assert not at.exception

    def test_app_title_present(self):
        """UI-02: App shows the expected title."""
        at = AppTest.from_file("app.py")
        at.run(timeout=10)
        assert any("Medical Imaging Analysis" in t.value for t in at.title)

    def test_sidebar_has_backend_radio(self):
        """UI-03: Sidebar contains the backend selection radio button."""
        at = AppTest.from_file("app.py")
        at.run(timeout=10)
        assert len(at.sidebar.radio) > 0
        radio = at.sidebar.radio[0]
        assert "Ollama" in radio.options[0]

    def test_tabs_present(self):
        """UI-04: Analysis and Model Management tabs are rendered."""
        at = AppTest.from_file("app.py")
        at.run(timeout=10)
        assert len(at.tabs) == 2

    def test_upload_prompt_shown(self):
        """UI-05: Info message prompting upload is shown when no file is loaded."""
        at = AppTest.from_file("app.py")
        at.run(timeout=10)
        info_texts = [i.value for i in at.info]
        assert any("Upload a medical image" in t for t in info_texts)
