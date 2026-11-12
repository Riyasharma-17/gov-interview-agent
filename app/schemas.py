from typing import Generic, Literal, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")

Confidence = Literal["high", "medium", "low"]
ReviewStatus = Literal["draft", "pending_review", "approved", "escalated"]


class TrackedField(BaseModel, Generic[T]):
    value: T | None = None
    source: str | None = None
    confidence: Confidence | None = None


class ConflictFlag(BaseModel):
    field_name: str
    claimed_value: str
    policy_excerpt: str
    reason: str


class SystemProfile(BaseModel):
    purpose: TrackedField[str] = Field(default_factory=TrackedField)
    data_types: TrackedField[list[str]] = Field(default_factory=TrackedField)
    owner: TrackedField[str] = Field(default_factory=TrackedField)
    approved_users: TrackedField[list[str]] = Field(default_factory=TrackedField)
    third_party_model: TrackedField[str] = Field(default_factory=TrackedField)
    status: ReviewStatus = "draft"
    conflicts: list[ConflictFlag] = Field(default_factory=list)