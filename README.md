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

## Key functions
- `load_environment`: optionally loads `.env` for local development.
- `ensure_api_key`: validates `OPENAI_API_KEY` before model calls.
- `build_llm`: constructs a `ChatOpenAI` client with optional temperature.
- `build_spec_extraction_chain`: wires extraction and transform prompts into a single runnable.
- `extract_specifications`: end-to-end wrapper that runs the chain for supplied text.

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

## Key functions
- `load_environment` / `ensure_api_key`: load env vars and enforce the API key is present.
- `build_llm`: constructs the `ChatOpenAI` client.
- `build_router_prompt`: defines the routing prompt for booker/info/unclear.
- `build_router_chain`: composes the prompt with the LLM and parser to emit a decision.
- `build_delegation_branch`: sets up handler branches and routing conditions.
- `build_coordinator_agent`: combines routing and delegation into one runnable chain.
- `booking_handler` / `info_handler` / `unclear_handler`: simulated downstream handlers used by the delegation branch.

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

# LangGraph Support Triage Demo

This example uses LangGraph to classify support messages and route them to FAQ, escalation, or fallback responses.

## How it works
- Classifier: `build_classifier_chain` labels each message as `faq`, `escalate`, or `fallback` using a prompt and `ChatOpenAI`.
- Routing: `build_graph` wires a `StateGraph` with a classify node and conditional edges to FAQ, escalation, or fallback nodes.
- Responses: FAQ questions flow through `build_faq_chain`; sensitive items go to an escalation stub; unknowns get a clarification request.
- Environment: `load_environment` loads `.env` when available; `ensure_api_key` validates `OPENAI_API_KEY` before building the LLM.
- CLI: `python3 langgraph_example.py` runs three demo messages and prints the intent and reply.

## Key functions
- `load_environment` / `ensure_api_key`: load env vars and enforce `OPENAI_API_KEY` is set.
- `build_llm`: constructs the `ChatOpenAI` client with optional temperature.
- `build_classifier_chain`: prompt+LLM+parser that emits `faq`/`escalate`/`fallback`.
- `build_faq_chain`: prompt+LLM+parser that returns concise FAQ answers.
- `build_graph`: assembles the LangGraph with classify → conditional routing → FAQ/escalate/fallback nodes and termination edges.
- `main`: runs the compiled graph against demo messages and logs intent plus reply.

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
   python3 langgraph_example.py
   ```
