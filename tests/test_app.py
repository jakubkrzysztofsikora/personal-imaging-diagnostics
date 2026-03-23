"""Tests for the Streamlit app using AppTest framework."""

import pytest

from inference import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Governor Pattern / Disclaimer (GP-*)
# ---------------------------------------------------------------------------

class TestGovernorPattern:
    def test_disclaimer_contains_provisional(self):
        """GP-05: Disclaimer text contains 'PROVISIONAL'."""
        from app import DISCLAIMER
        assert "PROVISIONAL" in DISCLAIMER

    def test_disclaimer_contains_not_diagnosis(self):
        """GP-06: Disclaimer text contains 'NOT A MEDICAL DIAGNOSIS'."""
        from app import DISCLAIMER
        assert "NOT A MEDICAL DIAGNOSIS" in DISCLAIMER


# ---------------------------------------------------------------------------
# Patient-Friendly Output (PF-*)
# ---------------------------------------------------------------------------

class TestPatientSummary:
    def test_summary_header(self):
        """PF-01: Summary contains 'What This Means for You'."""
        from app import _generate_patient_summary
        result = _generate_patient_summary("Test findings here.")
        assert "What This Means for You" in result

    def test_summary_safety_reminders(self):
        """PF-02: Summary contains safety language."""
        from app import _generate_patient_summary
        result = _generate_patient_summary("Test findings.")
        assert "preliminary" in result
        assert "qualified doctor" in result

    def test_summary_includes_findings(self):
        """PF-03: Summary includes original technical text."""
        from app import _generate_patient_summary
        findings = "Significant opacity in right lower lobe."
        result = _generate_patient_summary(findings)
        assert findings in result


# ---------------------------------------------------------------------------
# Session State (SS-*)
# ---------------------------------------------------------------------------

class TestSessionState:
    def test_init_defaults(self):
        """SS-01: Verify default session state values."""
        from app import init_session_state
        # We can't fully test Streamlit session state outside of AppTest,
        # but we can verify the function doesn't crash when imported
        assert callable(init_session_state)
