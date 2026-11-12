from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.llm import get_llm
from app.schemas import SystemProfile
from app.tools import flag_conflict, lookup_document, update_profile

TOOLS = [lookup_document, update_profile, flag_conflict]

SYSTEM_PROMPT = """You are an AI governance intake interviewer. Collect 5 fields for one AI system:
1. purpose (what the system does)
2. data_types (list of data categories processed, e.g. ["public docs", "customer PII"])
3. owner (accountable person/role)
4. approved_users (list of teams/roles allowed to use it)
5. third_party_model (exact model/vendor used)

Rules:
- TOOL-FIRST RULE: Never write a conversational reply until you have called `lookup_document`, saved EVERY stated field via `update_profile`, and flagged EVERY policy conflict via `flag_conflict`.
- Call `lookup_document` once at the start to read the policy; do not repeat identical queries.
- Batch all `update_profile` and `flag_conflict` calls together in one step.
- Policy violations to ALWAYS flag with `flag_conflict` (and save with `confidence="low"`):
  1. Any model other than enterprise Anthropic Claude or Google Gemini (e.g., OpenAI, GPT-4o, local/unapproved APIs) -> flag `third_party_model`.
  2. Any customer PII, emails, phone numbers, SSNs, payment cards, or medical/health records -> flag `data_types`.
  3. Broad access like "all-company", "everyone", or external contractors -> flag `approved_users`.
- Compliant fields: save via `update_profile` (`confidence="high"` if explicit, `"medium"` if implied).
- Only after all tool calls finish: ask for any remaining `None` fields (max 1-2 per turn), or summarize if all 5 are filled.

Current profile state:
{profile_json}
"""


def merge_profile(left: SystemProfile, right: SystemProfile) -> SystemProfile:
    merged = left.model_copy(deep=True)
    for field_name in ("purpose", "data_types", "owner", "approved_users", "third_party_model"):
        right_field = getattr(right, field_name)
        if right_field.value is not None:
            setattr(merged, field_name, right_field)

    seen = {(c.field_name, c.claimed_value) for c in merged.conflicts}
    for conflict in right.conflicts:
        if (conflict.field_name, conflict.claimed_value) not in seen:
            merged.conflicts.append(conflict)
            seen.add((conflict.field_name, conflict.claimed_value))

    if "pending_review" in (left.status, right.status):
        merged.status = "pending_review"
    else:
        merged.status = right.status
    return merged


class InterviewState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    profile: Annotated[SystemProfile, merge_profile]


def interview_node(state: InterviewState) -> dict:
    llm = get_llm(TOOLS)
    prompt = SYSTEM_PROMPT.format(profile_json=state["profile"].model_dump_json(indent=2))
    response = llm.invoke([SystemMessage(content=prompt), *state["messages"]])
    return {"messages": [response]}


def should_continue(state: InterviewState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("interview", interview_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "interview")
    graph.add_conditional_edges("interview", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "interview")
    return graph.compile()