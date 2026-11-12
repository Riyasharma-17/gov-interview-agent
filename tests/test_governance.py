import sqlite3
import pytest
from app.rules import evaluate_profile
from app.schemas import SystemProfile, TrackedField
from app.store import GovernanceStore


def _valid_profile() -> SystemProfile:
    return SystemProfile(
        purpose=TrackedField(value="Internal docs search", source="user", confidence="high"),
        data_types=TrackedField(value=["public docs"], source="user", confidence="high"),
        owner=TrackedField(value="Priya Nair (Engineering Manager)", source="user", confidence="high"),
        approved_users=TrackedField(value=["Platform Team"], source="user", confidence="high"),
        third_party_model=TrackedField(value="Google Gemini 2.5 Flash", source="user", confidence="high"),
    )


def test_rule_verdicts():
    assert evaluate_profile(_valid_profile()).verdict == "Yes"

    incomplete = _valid_profile()
    incomplete.owner = TrackedField()
    assert evaluate_profile(incomplete).verdict == "Unknown"

    violation = _valid_profile()
    violation.data_types = TrackedField(value=["customer PII"], source="user", confidence="high")
    res = evaluate_profile(violation)
    assert res.verdict == "No"
    assert any("PII" in r for r in res.reasons)


def test_audit_log_is_append_only_and_guards_transitions(tmp_path):
    store = GovernanceStore(tmp_path / "test.db")
    sub_id, _ = store.save_intake("DocBot", _valid_profile())

    with pytest.raises(ValueError, match="non-empty review note"):
        store.review_submission(sub_id, "approved", "alice", "   ")

    store.review_submission(sub_id, "approved", "alice", "Verified against policy.")
    with pytest.raises(ValueError, match="Invalid transition"):
        store.review_submission(sub_id, "draft", "alice", "Reverting.")

    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM audit_log")