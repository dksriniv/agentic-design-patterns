from __future__ import annotations

import math
import os
from typing import Callable, List, Literal, Tuple, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
    sources: List[str]


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
    """Creates a grounded FAQ answer using retrieved context."""
    prompt = ChatPromptTemplate.from_template(
        "You are a support agent. Use the context first; if it lacks the answer, say you are unsure briefly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {message}"
    )
    return prompt | llm | StrOutputParser()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_retriever() -> Callable[[str, int], Tuple[str, List[str]]]:
    """Create a simple in-memory retriever using OpenAI embeddings."""
    corpus = [
        "Reset password: click 'Forgot Password' on the login page and follow the email link.",
        "Update billing: visit the Billing page in settings to change card details.",
        "Refunds: refunds post within 5–7 business days after approval.",
        "Two-factor auth: enable it in Security settings for extra protection.",
    ]
    embeddings = OpenAIEmbeddings()
    corpus_vectors = embeddings.embed_documents(corpus)

    def retrieve(query: str, k: int = 3) -> Tuple[str, List[str]]:
        query_vec = embeddings.embed_query(query)
        scored = [
            (cosine_similarity(query_vec, vector), doc)
            for doc, vector in zip(corpus, corpus_vectors)
        ]
        top = sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]
        sources = [doc for _, doc in top]
        context = "\n\n".join(sources)
        return context, sources

    return retrieve


def build_graph(llm: ChatOpenAI, retrieve: Callable[[str, int], Tuple[str, List[str]]]):
    """Build a LangGraph that classifies and routes support messages with RAG-backed FAQ."""
    classifier = build_classifier_chain(llm)
    faq_chain = build_faq_chain(llm)

    def classify(state: SupportState) -> SupportState:
        raw_intent = classifier.invoke({"message": state["user_message"]}).strip()
        normalized = raw_intent.split()[0].lower() if raw_intent else ""
        intent = normalized if normalized in {"faq", "escalate", "fallback"} else "fallback"
        return {**state, "intent": intent}

    def answer_faq(state: SupportState) -> SupportState:
        context, sources = retrieve(state["user_message"])
        response = faq_chain.invoke(
            {"message": state["user_message"], "context": context}
        ).strip()
        return {**state, "response": response, "sources": sources}

    def escalate_ticket(state: SupportState) -> SupportState:
        message = (
            "Your request looks sensitive or complex. I've escalated it to a human "
            "specialist who will follow up shortly."
        )
        return {**state, "response": message}

    def fallback(state: SupportState) -> SupportState:
        message = "I couldn't determine the best path. Could you share more details?"
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
    retriever = build_retriever()
    support_graph = build_graph(llm, retriever)
    demo_messages = [
        "How do I reset my password?",
        "I need to update my credit card for billing.",
        "What is your refund timeline?",
    ]

    for message in demo_messages:
        result = support_graph.invoke({"user_message": message})
        print("\n--- SUPPORT RAG RUN ---")
        print(f"Message : {message}")
        print(f"Intent  : {result.get('intent')}")
        print(f"Reply   : {result.get('response')}")
        if sources := result.get("sources"):
            print("Sources :")
            for source in sources:
                print(f"- {source}")


if __name__ == "__main__":
    main()
