import json
import time
from pathlib import Path
from langchain_core.messages import HumanMessage
from app.agent import build_graph
from app.rules import evaluate_profile
from app.schemas import SystemProfile

CASES_PATH = Path(__file__).resolve().parent / "cases.json"


def run() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    graph = build_graph()

    planted_total = 0
    planted_caught = 0
    clean_total = 0
    clean_false_flags = 0

    for idx, case in enumerate(cases):
        if idx > 0:
            time.sleep(12)

        result = graph.invoke(
            {"messages": [HumanMessage(content=case["prompt"])], "profile": SystemProfile()},
            config={"recursion_limit": 20},
        )
        profile: SystemProfile = result["profile"]
        verdict_res = evaluate_profile(profile)
        flagged_fields = {c.field_name for c in profile.conflicts}
        expected_fields = set(case["expected_conflict_fields"])

        if expected_fields:
            planted_total += 1
            caught = expected_fields.issubset(flagged_fields)
            planted_caught += int(caught)
        else:
            clean_total += 1
            false_flag = len(flagged_fields) > 0
            clean_false_flags += int(false_flag)

        print(
            f"[{case['id']}] expected_verdict={case['expected_verdict']} actual_verdict={verdict_res.verdict} "
            f"expected_flags={sorted(expected_fields)} actual_flags={sorted(flagged_fields)}"
        )

    catch_rate = (planted_caught / planted_total * 100) if planted_total else 0.0
    false_flag_rate = (clean_false_flags / clean_total * 100) if clean_total else 0.0
    print("\n--- EVAL SUMMARY ---")
    print(f"Conflict detection rate: {planted_caught}/{planted_total} ({catch_rate:.1f}%)")
    print(f"False-flag rate:         {clean_false_flags}/{clean_total} ({false_flag_rate:.1f}%)")


if __name__ == "__main__":
    run()