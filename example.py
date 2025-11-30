from __future__ import annotations

import argparse
import os

try:
    # Optional helper to load variables from a local .env file during development.
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

DEFAULT_TEXT = (
    "The new laptop model features a 3.5 GHz octa-core processor, "
    "16GB of RAM, and a 1TB NVMe SSD."
)

PROMPT_EXTRACT = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{text_input}"
)

PROMPT_TRANSFORM = ChatPromptTemplate.from_template(
    (
        "Transform the following specifications into a JSON object with "
        "'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
    )
)


def load_environment() -> None:
    """Load environment variables from a .env file if python-dotenv is installed."""
    if load_dotenv:
        load_dotenv()


def ensure_api_key() -> None:
    """Provide a clearer error when the OpenAI API key is missing."""
    load_environment()
    if os.getenv("OPENAI_API_KEY"):
        return
    raise SystemExit(
        "Set the OPENAI_API_KEY environment variable to run this script. "
        "For local development you can keep it in a .env file."
    )


def build_llm(*, temperature: float = 0.0) -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    return ChatOpenAI(temperature=temperature)


def build_spec_extraction_chain(llm: ChatOpenAI) -> Runnable:
    """Create a runnable chain that extracts specs then converts to JSON."""
    extraction_chain = PROMPT_EXTRACT | llm | StrOutputParser()
    return (
        {"specifications": extraction_chain}
        | PROMPT_TRANSFORM
        | llm
        | StrOutputParser()
    )


def extract_specifications(
    text_input: str, *, temperature: float = 0.0
) -> str:
    """Run the full pipeline for a given input string."""
    ensure_api_key()
    llm = build_llm(temperature=temperature)
    chain = build_spec_extraction_chain(llm)
    return chain.invoke({"text_input": text_input})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hardware specs from text and return JSON."
    )
    parser.add_argument(
        "--text",
        "-t",
        default=DEFAULT_TEXT,
        help="Input text containing hardware specs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Language model sampling temperature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_result = extract_specifications(
        args.text,
        temperature=args.temperature,
    )
    print("\n--- Final JSON Output ---")
    print(final_result)


if __name__ == "__main__":
    main()
