# Tools/feedback_tool.py
import json
from pathlib import Path
from typing import Any

# Define your custom workflow state type
State = dict[str, Any]

def collect_and_store_feedback(state: State, json_path: str = "feedback_log.json") -> State:
    """
    Collects feedback (correct/useful) for each sub-question result.
    If any answer is marked incorrect or unhelpful, it prompts for a free-text explanation
    and saves the information to a JSON file.

    Args:
        state (State): LangGraph workflow state with questions and results.
        json_path (str): Path to the feedback log JSON file.

    Returns:
        State: Updated state with embedded feedback entries.
    """
    feedback_log = []
    feedback_file = Path(json_path)

    # Load existing feedback (if file exists)
    if feedback_file.exists():
        with open(feedback_file, "r", encoding="utf-8") as f:
            feedback_log = json.load(f)

    print("\nStarting feedback collection...\n")

    for idx, (question, result) in enumerate(zip(state["questions"], state["query_results"])):
        print(f"--- Sub‑question {idx + 1} ---")
        print(f"Q: {question}")
        print(f"A:\n{result}\n")

        correct = input("Was the answer factually correct? (yes/no): ").strip().lower()
        useful = input("Was the answer useful for your needs? (yes/no): ").strip().lower()

        feedback_entry = {
            "question": question,
            "answer": result,
            "correct": correct == "yes",
            "useful": useful == "yes"
        }

        if correct != "yes" or useful != "yes":
            explanation = input("Please explain what failed or was missing in the answer: ").strip()
            feedback_entry["explanation"] = explanation

        # Add entry to file log and optionally to state
        feedback_log.append(feedback_entry)
        state[f"feedback_{idx}"] = feedback_entry

    # Save all feedback to the JSON file
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(feedback_log, f, indent=2, ensure_ascii=False)
    return state
