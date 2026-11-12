import os
import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from app.agent import build_graph
from app.rules import evaluate_profile
from app.schemas import SystemProfile, TrackedField
from app.store import GovernanceStore

store = GovernanceStore()
graph = build_graph()


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def seed_demo_data() -> None:
    if store.list_submissions():
        return
    flagged = SystemProfile(
        purpose=TrackedField(value="Customer support email responder", source="turn_1", confidence="high"),
        data_types=TrackedField(value=["customer PII", "emails"], source="turn_2", confidence="low"),
        owner=TrackedField(value="Sarah Jenkins (Engineering Manager)", source="turn_2", confidence="high"),
        approved_users=TrackedField(value=["Support Team"], source="turn_3", confidence="high"),
        third_party_model=TrackedField(value="OpenAI GPT-4o", source="turn_1", confidence="low"),
    )
    store.save_intake("SupportMail-AI (Seeded Demo)", flagged, actor="seed_script")


def chat_turn(user_msg: str, history: list[dict], agent_state: dict):
    if not user_msg.strip():
        return "", history, agent_state, agent_state["profile"].model_dump_json(indent=2), ""

    state_in = {
        "messages": [*agent_state["messages"], HumanMessage(content=user_msg)],
        "profile": agent_state["profile"],
    }
    result = graph.invoke(state_in, config={"recursion_limit": 20})
    profile: SystemProfile = result["profile"]
    reply = extract_text(result["messages"][-1].content)

    updated_history = [
        *history,
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    new_state = {"messages": [*state_in["messages"], AIMessage(content=reply)], "profile": profile}
    verdict = evaluate_profile(profile)
    verdict_summary = f"Verdict: {verdict.verdict} | Status: {profile.status}\n" + "\n".join(f"- {r}" for r in verdict.reasons)
    return "", updated_history, new_state, profile.model_dump_json(indent=2), verdict_summary


def save_current_intake(system_name: str, agent_state: dict) -> str:
    if not system_name.strip():
        return "Error: Enter a system name before saving."
    sub_id, verdict = store.save_intake(system_name.strip(), agent_state["profile"])
    return f"Saved submission '{sub_id}' with status='{agent_state['profile'].status}' and verdict='{verdict.verdict}'."


def refresh_queue() -> tuple[list[list], list[str]]:
    rows = store.list_submissions()
    table = [[r["submission_id"], r["system_name"], r["status"], r["verdict"], r["updated_at"]] for r in rows]
    ids = [r["submission_id"] for r in rows]
    return table, ids


def load_submission(sub_id: str) -> tuple[str, list[list]]:
    if not sub_id:
        return "Select a submission ID.", []
    rows = [r for r in store.list_submissions() if r["submission_id"] == sub_id]
    profile_text = rows[0]["profile_json"] if rows else "Not found."
    logs = store.get_audit_log(sub_id)
    audit_rows = [[l["event_id"], l["actor"], l["action"], l["from_status"], l["to_status"], l["note"], l["created_at"]] for l in logs]
    return profile_text, audit_rows


def submit_review(sub_id: str, decision: str, reviewer: str, note: str) -> str:
    try:
        store.review_submission(sub_id, decision, reviewer or "compliance_reviewer", note)
        return f"Updated {sub_id} -> {decision}."
    except Exception as exc:
        return f"Error: {exc}"


def build_ui() -> gr.Blocks:
    seed_demo_data()
    init_state = {"messages": [], "profile": SystemProfile()}

    with gr.Blocks(title="AI Governance Interview Agent") as demo:
        gr.Markdown("## AI Governance Intake & Review Console")
        agent_state = gr.State(init_state)

        with gr.Tab("1. Intake Interview"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=380)
                    msg_box = gr.Textbox(placeholder="Describe your AI system (purpose, model, data, owner, users)...", label="Message")
                    send_btn = gr.Button("Send", variant="primary")
                with gr.Column(scale=1):
                    verdict_box = gr.Textbox(label="Deterministic Rule Verdict", lines=4, interactive=False)
                    profile_box = gr.Code(label="Live SystemProfile", language="json", value=SystemProfile().model_dump_json(indent=2))
                    sys_name = gr.Textbox(label="System Name (to save)", placeholder="e.g. SupportBot-v1")
                    save_btn = gr.Button("Save to Review Queue")
                    save_status = gr.Textbox(label="Queue Status", interactive=False)

            send_btn.click(chat_turn, [msg_box, chatbot, agent_state], [msg_box, chatbot, agent_state, profile_box, verdict_box])
            msg_box.submit(chat_turn, [msg_box, chatbot, agent_state], [msg_box, chatbot, agent_state, profile_box, verdict_box])
            save_btn.click(save_current_intake, [sys_name, agent_state], [save_status])

        with gr.Tab("2. Review Queue & Audit Log"):
            refresh_btn = gr.Button("Refresh Submissions")
            queue_table = gr.Dataframe(headers=["ID", "System", "Status", "Verdict", "Updated"], interactive=False)
            selected_id = gr.Dropdown(label="Select Submission ID", choices=[])
            selected_profile = gr.Code(label="Stored Profile JSON", language="json")

            with gr.Row():
                reviewer_box = gr.Textbox(label="Reviewer", value="lead_auditor")
                decision_box = gr.Dropdown(label="Decision", choices=["approved", "escalated"], value="approved")
                note_box = gr.Textbox(label="Required Audit Note", placeholder="Reason for approval or escalation...")
                review_btn = gr.Button("Record Decision", variant="primary")
            review_msg = gr.Textbox(label="Review Result", interactive=False)

            gr.Markdown("### Append-Only Audit Log")
            audit_table = gr.Dataframe(
                headers=["Event", "Actor", "Action", "From", "To", "Note", "Timestamp"],
                interactive=False,
            )

            def _refresh():
                table, ids = refresh_queue()
                return table, gr.Dropdown(choices=ids, value=ids[0] if ids else None)

            refresh_btn.click(_refresh, outputs=[queue_table, selected_id])
            selected_id.change(load_submission, [selected_id], [selected_profile, audit_table])
            review_btn.click(submit_review, [selected_id, decision_box, reviewer_box, note_box], [review_msg])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))