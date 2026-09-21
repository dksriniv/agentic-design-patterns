"""Run offline simulations or a live model through the same harness."""
import argparse
import json
import os
from harness.agents import LiveAgent, SCENARIOS, ScriptedAgent
from harness.runtime import Harness, Session
from harness.tools import FAQSearch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["scripted", "live"], default="scripted")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="account-recovery")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.mode == "scripted":
        print("SCRIPTED SIMULATION: actions and the clarification reply are predetermined.")
        agent = ScriptedAgent(args.scenario)
        request = SCENARIOS[args.scenario]
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("Live mode requires OPENAI_API_KEY. See README for setup.")
        try:
            agent = LiveAgent(args.model)
        except ImportError:
            parser.error("Install project dependencies with uv sync to use live mode.")
        print("LIVE MODE: model calls use your API account. Trace output includes messages and tool results.")
        request = input("Your support question: ").strip()
        if not request:
            parser.error("A support question is required")
    print("Request:", request)
    harness = Harness(agent, FAQSearch(), trace=lambda event: print(json.dumps(event, indent=2)))
    session = harness.run(Session(request=request, max_steps=args.max_steps))
    while session.status == "waiting_for_user":
        print("Agent:", session.pending_question)
        if args.mode == "scripted":
            reply = "I lost my phone."
            print("Simulated user:", reply)
        else:
            reply = input("You: ").strip()
            if not reply:
                print("Please enter a reply.")
                continue
        session = harness.resume(session, reply)
    print("Final status:", session.status)
    if session.answer:
        print("Answer:", session.answer)
        print("Sources:", ", ".join(session.sources) or "none")
    return 1 if session.status in {"agent_error", "limit_reached"} else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print("\nSession ended by user. In-memory progress is not saved.")
        raise SystemExit(130)
