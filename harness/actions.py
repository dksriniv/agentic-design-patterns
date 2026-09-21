"""The small contract shared by scripted and live agents."""
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Action:
    name: str
    arguments: Dict[str, Any]


def validate(raw: Any) -> Action:
    if not isinstance(raw, dict) or set(raw) != {"action", "arguments"}:
        raise ValueError("Expected an object with action and arguments keys.")
    name, args = raw["action"], raw["arguments"]
    fields = {
        "search_faq": {"query"},
        "ask_user": {"question"},
        "finish": {"answer", "sources", "outcome"},
    }
    if not isinstance(name, str) or name not in fields:
        raise ValueError("Unknown action. Choose search_faq, ask_user, or finish.")
    if not isinstance(args, dict) or set(args) != fields[name]:
        raise ValueError("Arguments do not match the action schema.")
    text_field = {"search_faq": "query", "ask_user": "question", "finish": "answer"}[name]
    if not isinstance(args[text_field], str) or not args[text_field].strip():
        raise ValueError(f"{text_field} must be a nonempty string.")
    if name == "finish":
        if args["outcome"] not in ("answered", "insufficient_information"):
            raise ValueError("Invalid finish outcome.")
        if not isinstance(args["sources"], list) or not all(isinstance(s, str) for s in args["sources"]):
            raise ValueError("sources must be a list of document IDs.")
    return Action(name, args)
