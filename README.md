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

# LangGraph RAG Support Demo

This example blends LangGraph routing with a small RAG pipeline to ground FAQ answers before replying.

## How it works
- Classifier: `build_classifier_chain` labels each message as `faq`, `escalate`, or `fallback`.
- Retrieval: `build_retriever` embeds a short FAQ corpus with `OpenAIEmbeddings` and selects the top matches via cosine similarity.
- Routing: `build_graph` wires classify → conditional routing → FAQ/escalate/fallback nodes and termination edges.
- Responses: `answer_faq` pulls retrieved context into the FAQ prompt; `escalate_ticket` and `fallback` handle sensitive or unclear cases.
- Environment: `load_environment` loads `.env` when available; `ensure_api_key` validates `OPENAI_API_KEY` before building the LLM.
- CLI: `python3 langgraph_rag_example.py` runs three demo messages, printing intent, reply, and retrieved sources.

## Key functions
- `load_environment` / `ensure_api_key`: load env vars and enforce `OPENAI_API_KEY` is set.
- `build_llm`: constructs the `ChatOpenAI` client with optional temperature.
- `build_classifier_chain`: prompt+LLM+parser that emits `faq`/`escalate`/`fallback`.
- `build_retriever`: creates a simple embedding-backed retriever plus cosine similarity helper.
- `build_faq_chain`: prompt+LLM+parser that returns concise FAQ answers grounded in context.
- `build_graph`: assembles the LangGraph with classify → RAG FAQ/escalate/fallback nodes and termination edges.
- `main`: runs the compiled graph against demo messages and logs intent, reply, and sources.

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
   python3 langgraph_rag_example.py
   ```

# Plain Python Agent Harness (Learning Example)

This example makes the execution loop visible. An agent proposes an action; the
harness validates it, runs the tool, records the result, and decides whether the
session can continue. The live model can change its next action after reading
results. The scripted agent simulates those decisions for repeatable teaching.

## Start offline

No dependencies or API key are required for scripted mode (Python 3.9+):

```sh
python3 harness_example.py --mode scripted --scenario direct-answer
python3 harness_example.py --mode scripted --scenario account-recovery
python3 harness_example.py --mode scripted --scenario unsuccessful-search
```

`account-recovery` searches, pauses for clarification, receives the **scripted**
reply “I lost my phone,” searches again, then answers. The pause is real harness
state, but the CLI supplies the predetermined reply. Scripted mode is a simulation,
not an autonomous agent, and does not interpret arbitrary user replies.

`unsuccessful-search` tries two queries, finds no evidence, then finishes with an
`insufficient_information` outcome. To see a controlled budget stop:

```sh
python3 harness_example.py --mode scripted --scenario account-recovery --max-steps 2
```

This ends with `limit_reached` and exit code 1. The harness will not ask a question
when no action budget remains to process the reply.

## Use a live model

Install the existing project dependencies with `uv sync`, set `OPENAI_API_KEY`
in your environment or a local `.env` file, then run:

```sh
uv run python harness_example.py --mode live
uv run python harness_example.py --mode live --model gpt-4.1-mini --max-steps 8
```

The default model is `gpt-4.1-mini`; `OPENAI_MODEL` or `--model` can override it.
Use a model available to your account. Live mode uses your API account and asks
for your support question and any clarification replies in the terminal. Try
“How do I regain access to my account?” and then “I lost my phone.” The live
model's exact sequence can vary; scripted mode is the repeatable baseline.

Both modes search the same tiny fictional knowledge base with local word matching.
Only live mode imports the existing LangChain model client. Python owns the loop;
there is no LangGraph or framework agent executor here.

## Follow the execution

```text
request -> model/script proposes action -> validate -> execute -> record
                     ^                                  |
                     +----------- tool result ----------+
                     +----------- user reply (after pause)
```

Read the files in this order:

1. `harness/actions.py`: the three action contracts.
2. `harness/tools.py` and `harness/knowledge_base.json`: offline retrieval.
3. `harness/runtime.py`: session state, execution, pausing and resuming.
4. `harness/agents.py`: interchangeable scripted and live action producers.
5. `harness_example.py`: CLI and human interaction.

Every trace event includes the proposed action, input, result, and remaining
budget. It does not request or print private model reasoning. Traces do include
user messages and retrieved content; use fictional inputs for these exercises.

The harness enforces:

- A five-attempt default budget shared across clarification turns. Invalid model
  output and failed calls also consume attempts. A user reply does not.
- Exact action names and argument shapes; malformed live JSON becomes feedback
  for the next attempt.
- No repeat execution of an identical search after whitespace/case normalization.
- Every cited ID must have been retrieved. An `answered` outcome needs a source.
- Tool errors become observations. Provider errors stop with `agent_error` without
  exposing the provider exception body. Live calls have a 30-second client timeout
  and automatic client retries disabled.

Final statuses are `answered`, `insufficient_information`, `limit_reached`, and
`agent_error`. `waiting_for_user` is a paused state, not completion. User interruption
exits with code 130. Progress exists only in memory and is lost on exit.

## Verify and experiment

```sh
python3 -m unittest discover -s tests -v
```

The offline tests cover normal completion, clarification and budget preservation,
empty results, invalid actions, invalid citations, duplicate queries, and failures.
They do not measure live-model answer quality. Citation checks establish that a
source was retrieved, **not** that the source supports every claim. Manually compare
live answers with the displayed documents and record whether the answer is supported,
whether clarification was useful, and how many actions were needed.

Exercises: change a scripted action to an unknown tool, cite a nonexistent document,
or reduce the step budget. Predict the trace, then run it. Next, add a document and
compare scripted and live behavior. Persisted sessions, embeddings, separate token
budgets, and semantic answer evaluation are deliberately later lessons.
