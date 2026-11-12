from typing import Literal
from pydantic import BaseModel
from app.schemas import SystemProfile

Verdict = Literal["Yes", "No", "Unknown"]

APPROVED_MODELS = ("claude", "anthropic", "gemini", "google")
RESTRICTED_DATA_KEYWORDS = ("pii", "email", "phone", "ssn", "payment", "card", "medical", "health")
DISALLOWED_USER_KEYWORDS = ("all-company", "all company", "everyone", "public", "contractor")
REQUIRED_FIELDS = ("purpose", "data_types", "owner", "approved_users", "third_party_model")


class VerdictResult(BaseModel):
    verdict: Verdict
    reasons: list[str]


def evaluate_profile(profile: SystemProfile) -> VerdictResult:
    reasons: list[str] = []

    for conflict in profile.conflicts:
        reasons.append(f"Flagged conflict on '{conflict.field_name}': {conflict.reason}")

    model_val = (profile.third_party_model.value or "").lower()
    if model_val and not any(allowed in model_val for allowed in APPROVED_MODELS):
        reasons.append(f"Unapproved model '{profile.third_party_model.value}'.")

    for dtype in profile.data_types.value or []:
        if any(kw in dtype.lower() for kw in RESTRICTED_DATA_KEYWORDS):
            reasons.append(f"Restricted data type '{dtype}' requires DPO sign-off.")

    for user_group in profile.approved_users.value or []:
        if any(kw in user_group.lower() for kw in DISALLOWED_USER_KEYWORDS):
            reasons.append(f"Disallowed access scope '{user_group}'.")

    if reasons:
        return VerdictResult(verdict="No", reasons=reasons)

    missing_or_low = []
    for field_name in REQUIRED_FIELDS:
        field = getattr(profile, field_name)
        if field.value is None or (isinstance(field.value, list) and len(field.value) == 0):
            missing_or_low.append(f"Missing required field '{field_name}'.")
        elif field.confidence == "low":
            missing_or_low.append(f"Low confidence on '{field_name}'; needs verification.")

    if missing_or_low:
        return VerdictResult(verdict="Unknown", reasons=missing_or_low)

    return VerdictResult(verdict="Yes", reasons=["All 5 governance checks passed with no policy conflicts."])