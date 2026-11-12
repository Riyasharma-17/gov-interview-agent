from pathlib import Path
from typing import Annotated, Any
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import ValidationError
from app.schemas import Confidence, ConflictFlag, SystemProfile, TrackedField

POLICY_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_policy.md"
LIST_FIELDS = {"data_types", "approved_users"}


@tool
def lookup_document(query: str) -> str:
    """Search the company AI governance policy for rules relevant to the query."""
    sections = POLICY_PATH.read_text(encoding="utf-8").split("\n## ")
    terms = {w.lower().strip(".,()") for w in query.split() if len(w) > 2}

    scored = []
    for raw in sections:
        section = raw if raw.startswith("## ") else f"## {raw}"
        words = {w.lower().strip(".,()") for w in section.split()}
        overlap = len(terms & words)
        if overlap > 0:
            scored.append((overlap, section.strip()))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return POLICY_PATH.read_text(encoding="utf-8").strip()
    return "\n\n".join(text for _, text in scored[:2])


@tool
def update_profile(
    field_name: str,
    value: str | list[str],
    source: str,
    confidence: Confidence,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command | str:
    """Save a verified field into the AI system profile."""
    profile: SystemProfile = state["profile"].model_copy(deep=True)
    if field_name not in SystemProfile.model_fields or field_name in {"status", "conflicts"}:
        return f"Error: '{field_name}' is not an editable profile field."

    if field_name in LIST_FIELDS and isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]

    try:
        tracked = TrackedField(value=value, source=source, confidence=confidence)
        setattr(profile, field_name, tracked)
        SystemProfile.model_validate(profile.model_dump())
    except ValidationError as exc:
        return f"Validation error for '{field_name}': {exc.errors()[0]['msg']}"

    msg = ToolMessage(content=f"Updated {field_name}.", tool_call_id=tool_call_id)
    return Command(update={"profile": profile, "messages": [msg]})


@tool
def flag_conflict(
    field_name: str,
    claimed_value: str,
    policy_excerpt: str,
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Flag a contradiction between a user's claim and the company policy for human review."""
    profile: SystemProfile = state["profile"].model_copy(deep=True)
    flag = ConflictFlag(
        field_name=field_name,
        claimed_value=claimed_value,
        policy_excerpt=policy_excerpt,
        reason=reason,
    )
    profile.conflicts.append(flag)
    profile.status = "pending_review"

    msg = ToolMessage(
        content=f"Conflict flagged on '{field_name}'. Status set to pending_review.",
        tool_call_id=tool_call_id,
    )
    return Command(update={"profile": profile, "messages": [msg]})