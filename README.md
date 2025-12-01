# Prompt Chaining Demo

This project shows a simple LangChain pipeline that:
- Extracts hardware specs from free text.
- Transforms them into a JSON object (`cpu`, `memory`, `storage`).
- Loads your `OPENAI_API_KEY` from the environment (optionally via `.env`).

## How it works
- Prompts: `PROMPT_EXTRACT` pulls specs from input text; `PROMPT_TRANSFORM` reshapes that output into JSON.
- Chain: `build_spec_extraction_chain` composes the prompts with `ChatOpenAI` and `StrOutputParser` to create a runnable that maps `text_input` ➜ specs ➜ JSON string.
- Environment: `load_environment` uses `python-dotenv` (if installed) to load `.env`, then `ensure_api_key` validates `OPENAI_API_KEY` before calling the model.
- CLI: `python3 example.py --text "The server has a 3.2GHz CPU, 32GB RAM, 2TB SSD"` runs the chain and prints the JSON. Use `--temperature` to adjust sampling.

## Setup
1) Add your key to `.env` (already gitignored):
   ```
   OPENAI_API_KEY=sk-...
   ```
2) Install deps (uv or pip):
   - `uv sync` (preferred)  
   - or `pip install -e .`
3) Run:
   ```
   python3 example.py
   python3 example.py --text "Custom spec text" --temperature 0.2
   ```

# Router Pattern Demo

This example shows a coordinator router that decides which specialist handler should process a request (booking, info, or unclear).

## How it works
- Router: `build_router_chain` uses a prompt plus `ChatOpenAI` to output `booker`, `info`, or `unclear`.
- Delegation: `build_delegation_branch` dispatches to `booking_handler`, `info_handler`, or `unclear_handler` via `RunnableBranch`.
- Environment: `load_environment` loads `.env` when available; `ensure_api_key` validates `OPENAI_API_KEY` before building the LLM.
- CLI: `python3 router_pattern.py` invokes the coordinator on a few demo requests and prints the results.

## Setup
1) Add your key to `.env` (already gitignored):
   ```
   OPENAI_API_KEY=sk-...
   ```
2) Install deps (uv or pip):
   - `uv sync` (preferred)  
   - or `pip install -e .`
3) Run:
   ```
   python3 router_pattern.py
   ```
