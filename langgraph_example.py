from __future__ import annotations

import os
from typing import Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

try:
    # Optional helper to load variables from a local .env file during development.
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class SupportState(TypedDict, total=False):
    user_message: str
    intent: Literal["faq", "escalate", "fallback"]
    response: str


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


def build_classifier_chain(llm: ChatOpenAI):
    """Classifies whether a support message is FAQ, needs escalation, or fallback."""
    prompt = ChatPromptTemplate.from_template(
        """Classify the user message and respond with one word:
- 'faq' for routine, answerable questions.
- 'escalate' for billing/account/security issues.
- 'fallback' when unsure.
Message: {message}"""
    )
    return prompt | llm | StrOutputParser()


def build_faq_chain(llm: ChatOpenAI):
    """Creates a short FAQ-style answer."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful support agent. Provide a concise answer to: {message}"
    )
    return prompt | llm | StrOutputParser()


def build_graph(llm: ChatOpenAI):
    """Build a LangGraph that classifies and routes support messages."""
    classifier = build_classifier_chain(llm)
    faq_chain = build_faq_chain(llm)

    def classify(state: SupportState) -> SupportState:
        intent = classifier.invoke({"message": state["user_message"]}).strip().lower()
        return {**state, "intent": intent}

    def answer_faq(state: SupportState) -> SupportState:
        response = faq_chain.invoke({"message": state["user_message"]}).strip()
        return {**state, "response": response}

    def escalate_ticket(state: SupportState) -> SupportState:
        message = (
            "Your request looks sensitive or complex. I've escalated it to a human "
            "specialist who will follow up shortly."
        )
        return {**state, "response": message}

    def fallback(state: SupportState) -> SupportState:
        message = (
            "I couldn't determine the best path. Could you share more details?"
        )
        return {**state, "response": message}

    graph = StateGraph(SupportState)
    graph.add_node("classify", classify)
    graph.add_node("answer_faq", answer_faq)
    graph.add_node("escalate_ticket", escalate_ticket)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: state["intent"],
        {
            "faq": "answer_faq",
            "escalate": "escalate_ticket",
            "fallback": "fallback",
            "__default__": "fallback",
        },
    )
    graph.add_edge("answer_faq", END)
    graph.add_edge("escalate_ticket", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def main() -> None:
    llm = build_llm()
    support_graph = build_graph(llm)
    demo_messages = [
        "How do I reset my password?",
        "My credit card was charged twice, help.",
        "Tell me something interesting.",
    ]

    for message in demo_messages:
        result = support_graph.invoke({"user_message": message})
        print("\n--- SUPPORT RUN ---")
        print(f"Message : {message}")
        print(f"Intent  : {result.get('intent')}")
        print(f"Reply   : {result.get('response')}")


if __name__ == "__main__":
    main()
