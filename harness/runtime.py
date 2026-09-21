"""The harness owns execution; agents only propose actions."""
from dataclasses import dataclass, field
from typing import Any, Dict, List
from .actions import validate


@dataclass
class Session:
    request: str
    max_steps: int = 5
    attempts: int = 0
    status: str = "running"
    events: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: Dict[str, Any] = field(default_factory=dict)
    searches: set = field(default_factory=set)
    pending_question: str = ""
    answer: str = ""
    sources: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

    @property
    def steps_remaining(self):
        return self.max_steps - self.attempts


class Harness:
    def __init__(self, agent, search, trace=lambda event: None):
        self.agent, self.search, self.trace = agent, search, trace

    def record(self, session, action, result):
        event = {"step": session.attempts, "action": action, "result": result,
                 "steps_remaining": session.steps_remaining}
        session.events.append(event)
        self.trace(event)

    def resume(self, session, reply):
        if session.status != "waiting_for_user":
            raise ValueError("Session is not waiting for a reply.")
        if not reply.strip():
            raise ValueError("Reply must not be empty.")
        self.record(session, {"action": "user_reply"}, {"reply": reply})
        session.pending_question = ""
        session.status = "running"
        return self.run(session)

    def run(self, session):
        if session.status != "running":
            return session
        while session.steps_remaining > 0:
            session.attempts += 1  # Invalid output and failed calls consume budget too.
            raw = None
            try:
                raw = self.agent.choose_action(session)
            except Exception as exc:
                # Avoid printing provider exception bodies, which may contain sensitive data.
                session.status = "agent_error"
                self.record(session, None, {"error": "agent_error", "type": type(exc).__name__})
                return session
            try:
                action = validate(raw)
                args = action.arguments
                if action.name == "search_faq":
                    query = " ".join(args["query"].lower().split())
                    if query in session.searches:
                        result = {"error": "repeated_search", "message": "Try a different query, ask the user, or finish."}
                    else:
                        session.searches.add(query)
                        try:
                            result = self.search.search(args["query"])
                            for doc in result["documents"]:
                                session.retrieved[doc["id"]] = doc
                        except Exception:
                            result = {"error": "tool_error", "message": "Search failed. Try another action."}
                elif action.name == "ask_user":
                    if session.steps_remaining == 0:
                        result = {"error": "no_resume_budget", "message": "No steps remain to process a reply."}
                    else:
                        session.pending_question = args["question"]
                        session.status = "waiting_for_user"
                        result = {"question": args["question"]}
                else:
                    if any(source not in session.retrieved for source in args["sources"]):
                        raise ValueError("Every cited source must have been retrieved in this session.")
                    if args["outcome"] == "answered" and not args["sources"]:
                        raise ValueError("An answered outcome requires at least one retrieved source.")
                    session.answer, session.sources = args["answer"], args["sources"]
                    session.status = args["outcome"]
                    result = dict(args)
                self.record(session, raw, result)
            except ValueError as exc:
                self.record(session, raw, {"error": "invalid_action", "message": str(exc)})
            if session.status != "running":
                return session
        session.status = "limit_reached"
        self.record(session, None, {"status": "limit_reached"})
        return session
