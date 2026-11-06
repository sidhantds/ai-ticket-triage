"""Tests for the AI ticket triage app.

Run from the repository root with:  pytest -v
"""
import importlib
import sys
import warnings
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def app():
    """Import app.py as a module (Streamlit runs in bare mode, UI calls are no-ops)."""
    warnings.filterwarnings("ignore")
    return importlib.import_module("app")


# ---------------------------------------------------------------- app startup
def test_app_starts_without_errors():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert "AI Ticket Operations" in [t.value for t in at.title]


# ---------------------------------------------------------------- text cleaning
def test_clean_text_normalizes_noise(app):
    cleaned = app.clean_text("Dear <name>, our SERVERS are crashing!! Error 404 on www.site.com")
    assert cleaned == "server crashing error"


def test_clean_text_keeps_negation(app):
    assert "not" in app.clean_text("The payment is not working").split()


# ---------------------------------------------------------------- PII masking
def test_redact_pii_masks_email_phone_and_ids(app):
    text = "Mail john.doe@example.com or call 555-123-4567 about account 12345678"
    redacted = app.redact_pii(text)
    assert "john.doe@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "12345678" not in redacted


# ---------------------------------------------------------------- input validation
@pytest.mark.parametrize(
    "subject, body",
    [("Subject", ""), ("Subject", "   "), ("Subject", "too short"), ("", "x" * 10_001)],
    ids=["empty", "whitespace", "too-short", "too-long"],
)
def test_validate_ticket_rejects_bad_input(app, subject, body):
    with pytest.raises(ValueError):
        app.validate_ticket(subject, body)


def test_validate_ticket_accepts_normal_ticket(app):
    assert "refund" in app.validate_ticket("Billing", "Please refund the duplicate charge.")


# ---------------------------------------------------------------- SLA mapping
def test_sla_mapping(app):
    assert app.SLA_HOURS == {"high": 1, "medium": 8, "low": 24}


# ---------------------------------------------------------------- prediction
def test_prediction_output_schema(app):
    result = app.analyze_ticket("Our production server is down and customers cannot log in.")
    expected_keys = {
        "category", "category_confidence", "priority", "priority_confidence",
        "sla_hours", "status", "key_terms", "top_categories",
    }
    assert expected_keys <= set(result)
    assert result["category"] in set(app.category_model.classes_)
    assert result["priority"] in {"high", "medium", "low"}
    assert 0 <= result["category_confidence"] <= 100
    assert result["sla_hours"] == app.SLA_HOURS[result["priority"]]
    assert len(result["top_categories"]) == 3
    confidences = [conf for _, conf in result["top_categories"]]
    assert confidences == sorted(confidences, reverse=True)


def test_status_follows_confidence_threshold(app):
    result = app.analyze_ticket("hello")
    should_route = result["category_confidence"] >= app.CONFIDENCE_THRESHOLD * 100
    assert result["status"] == ("Auto-Routed" if should_route else "Needs Human Review")


def test_billing_ticket_is_routed_to_billing(app):
    result = app.analyze_ticket(
        "I was charged twice for my subscription invoice. Please refund the duplicate billing charge."
    )
    assert result["category"] == "Billing and Payments"
    assert result["status"] == "Auto-Routed"
    assert {"billing", "charge"} & set(result["key_terms"])
