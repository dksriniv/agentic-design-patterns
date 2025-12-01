from __future__ import annotations

import os
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI

try:
    # Optional helper to load variables from a local .env file during development.
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

Handler = Callable[[str], str]


def load_environment() -> None:
    """Load environment variables from a .env file if python-dotenv is installed."""
    if load_dotenv:
        load_dotenv()


def ensure_api_key(env_var: str = "OPENAI_API_KEY") -> None:
    """Ensure the LLM has credentials before we try to use it."""
    load_environment()
    if os.getenv(env_var):
        return
    raise SystemExit(
        f"Set the {env_var} environment variable to run this script. "
        "For local development you can keep it in a .env file."
    )


def build_llm(*, temperature: float = 0.0) -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    ensure_api_key()
    return ChatOpenAI(temperature=temperature)


# --- Simulated sub-agent handlers -------------------------------------------------
def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request."""
    return f"Booking handler processed request: '{request}'. Result: simulated booking action."


def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request."""
    return f"Info handler processed request: '{request}'. Result: simulated information retrieval."


def unclear_handler(request: str) -> str:
    """Handles requests that could not be delegated."""
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


# --- Router and delegation chain --------------------------------------------------
def build_router_prompt() -> ChatPromptTemplate:
    """Prompt that decides which specialist handler should process the request."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Analyze the user's request and determine which specialist handler should process it.
- If the request is related to booking flights or hotels, output 'booker'.
- For all other general information questions, output 'info'.
- If the request is unclear or doesn't fit either category, output 'unclear'.
ONLY output one word: 'booker', 'info', or 'unclear'.""",
            ),
            ("user", "{request}"),
        ]
    )


def build_router_chain(llm: ChatOpenAI) -> Runnable:
    """Chain that produces the routing decision."""
    return build_router_prompt() | llm | StrOutputParser()


def build_delegation_branch() -> RunnableBranch:
    """Branch that dispatches to the correct handler based on the decision."""

    def make_branch(handler: Handler) -> Runnable:
        return RunnablePassthrough.assign(
            output=lambda x: handler(x["request_text"])
        )

    return RunnableBranch(
        (lambda x: x["decision"].strip().lower() == "booker", make_branch(booking_handler)),
        (lambda x: x["decision"].strip().lower() == "info", make_branch(info_handler)),
        make_branch(unclear_handler),
    )


def build_coordinator_agent(llm: ChatOpenAI) -> Runnable:
    """Combine routing and delegation into a single runnable."""
    router_chain = build_router_chain(llm)
    delegation_branch = build_delegation_branch()
    return (
        {
            "decision": router_chain,
            "request_text": lambda x: x["request"],
        }
        | delegation_branch
        | (lambda x: x["output"])
    )


# --- Example usage ----------------------------------------------------------------
def main() -> None:
    llm = build_llm()
    coordinator_agent = build_coordinator_agent(llm)

    demo_requests = [
        ("booking", "Book me a flight to London."),
        ("info", "What is the capital of Italy?"),
        ("unclear", "Tell me about quantum physics."),
    ]

    for label, text in demo_requests:
        result = coordinator_agent.invoke({"request": text})
        print(f"\n--- {label.upper()} REQUEST ---")
        print(f"Final result: {result}")


if __name__ == "__main__":
    main()
