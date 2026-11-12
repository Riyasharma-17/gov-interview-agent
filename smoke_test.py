from langchain_core.messages import HumanMessage
from app.agent import build_graph
from app.schemas import SystemProfile

app = build_graph()
state = {
    "messages": [
        HumanMessage(
            content="We built a support bot using OpenAI GPT-4o that reads customer PII and emails. Owner is Sarah (Engineering Manager), used by Support Team."
        )
    ],
    "profile": SystemProfile(),
}

for step in app.stream(state, config={"recursion_limit": 20}, stream_mode="updates"):
    for node_name, output in step.items():
        print(f"--- NODE: {node_name} ---")
        if "messages" in output:
            last_msg = output["messages"][-1]
            if getattr(last_msg, "tool_calls", None):
                print("TOOL CALLS:", [(tc["name"], tc["args"]) for tc in last_msg.tool_calls])
            else:
                print("CONTENT:", last_msg.content)
        if "profile" in output:
            print("STATUS:", output["profile"].status)
            print("CONFLICTS:", [c.field_name for c in output["profile"].conflicts])