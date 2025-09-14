# 📘 Project Best Practices

## 1. Project Purpose
A Python 3.12+ local-LLM-driven, agent-based simulation set in a medieval village. Multiple autonomous agents perceive their surroundings, plan activities, move on a grid-based world, converse, and reflect on experiences. The system leverages prompt templates (Jinja2) and a local llama.cpp model wrapper to generate schedules, microtasks, conversations, and reflective thoughts, while maintaining short- and long-term memory structures for retrieval and reasoning.

## 2. Project Structure
- Top-level entrypoint
  - `main.py` – Bootstraps the simulation: loads env, constructs model/world/agents, and runs the tick loop.
- Agent modules (LLM-driven decisions and cognition)
  - `agents/`
    - `perceive.py` – Gathers nearby tile events, logs chats and significant events, updates spatial memory.
    - `planner.py` – Modular planning pipeline: day blocks → hourly schedule → next action; event reaction; object interaction; microtasks; identity revision; chat decisions.
    - `executor.py` – Deterministic movement, path planning (Dijkstra), event placement on tiles, chat arrival/handling, subtask-to-memory logging.
    - `reflection.py` – Triggers periodic reflection, generates focal points and insights, stores new thoughts with evidence links.
    - `retrieval.py` – Context retrieval: vector-based ranking using pseudo-embeddings with recency/importance weights and keyword fallbacks.
    - `chat_manager.py`, `identity.py`, `event_triple_generator.py`, `inner_thought_generator.py` – Prompted utilities for conversation, identity/status refinement, triple extraction, and inner thoughts.
- Core simulation and infrastructure
  - `core/`
    - `agent_action.py` – Dataclasses: Event, AgentAction (with chat/path/subtasks), helpers for timing/summary.
    - `agent.py` – Agent state: identity, memories, spatial memory, and serialized "state" for prompts.
    - `world.py` – World timekeeping and global state; coordinates agents.
    - `tile.py` – Tile representation with events and collision.
    - `tile_manager.py` – Grid operations, address indexing (world:sector:arena:object), nearby queries, event attach/remove, finders.
    - `logger.py` – Logging setup used across modules.
    - `setup/`
      - `world_builder.py` – Generates the medieval village map with sectors/arenas/objects and gravel paths.
      - `agent_builder.py` – Constructs personas, seeds memories and initial thoughts.
- Memory abstractions
  - `memory/`
    - `short_term_memory.py` – Daily schedule, current action/time, reflection thresholds and weights.
    - `long_term_memory.py` – MemoryNode model, event/thought/chat indexing, keyword maps, pseudo-embedding storage.
    - `spatial_memory.py` – Known sectors/arenas/objects per world.
- Model and prompts
  - `models/`
    - `local_model.py` – llama.cpp wrapper with context management, allowed-strings grammar, env-based config.
    - `embeddings.py` – Deterministic pseudo-embeddings for ranking and tests.
  - `prompts/` – Jinja2 templates for all LLM interactions (planning, microtasks, conversations, reflection, etc.).
- Other
  - `requirements.txt`, `.env`, `.github/`, `.vscode/`, `README.md`, `Dockerfile`, `scripts/`, example prompt/code directories.

Conventions and key concepts:
- Address schema: `world:sector:arena:object` with graceful degradation for missing parts; special markers like `<waiting> x y` and `<random>`.
- Strong separation between deterministic execution (executor/world/tiles) and model-driven choices (agents/* + prompts/).
- Interfaces in `interfaces/` to formalize planner/executor contracts.

## 3. Test Strategy
Current status: no tests are present. Adopt the following strategy with Pytest:
- Framework: Pytest with coverage (`pytest --cov`); place tests in `tests/` mirroring module layout.
- Unit tests (priority targets)
  - `core.tile_manager` – bounds, collision, address indexing, finders, nearby queries.
  - `core.agent_action` – time math (start/end), chat timing, is_chat_finished edge cases.
  - `agents.executor` – pathfinding (Dijkstra), arrival behaviors, chat start conditions, subtask logging.
  - `agents.planner` – schedule parsing (day blocks, hourly), cross‑midnight handling, microtasks text parsing, option picking.
  - `agents.perceive` – selection/filtering of events, significance scoring fallbacks, spatial memory updates.
  - `memory.long_term_memory` – indexing, expiration, retrieval by keywords; uniqueness and ordering.
  - `agents.retrieval` – combined scoring normalization, top-N selection, idle filtering.
- Integration tests
  - Single-tick end-to-end: world setup → perceive → retrieve → plan → execute → perceive → reflect, asserting state transitions.
  - Chat meeting flow: apply reaction "chat with X" → move to adjacency → chat timing and logs.
- Mocking guidelines
  - Stub `LocalModel.generate` to deterministic functions per template to avoid LLM nondeterminism.
  - Use fixed seeds for any random selection in executor (e.g., `<random>` address degradation).
  - Replace `models.embeddings.get_embedding` with a fixed vector or simple hash for reproducibility (already deterministic).
- Naming
  - Test files: `test_<module>.py`; test functions: `test_<behavior>`. Use fixtures for world/agents (e.g., `village_world`, `farmer_agent`).
- Coverage targets
  - Aim for 80%+ in core deterministic modules; LLM prompt modules validated via contract tests and strict stubs.

## 4. Code Style
- Language and typing
  - Python 3.12+; use type hints consistently for function signatures and dataclasses.
  - Prefer dataclasses for simple state holders (`Event`, `AgentAction`, etc.).
- Naming conventions
  - Modules/files: `snake_case.py`.
  - Classes: `PascalCase`.
  - Functions/methods/variables: `snake_case`.
  - Prompts: descriptive names by intent (e.g., `generate_focal_points.txt`, `microtasks.txt`).
- Prompted generation
  - Keep templates minimal and structured; apply `allowed_strings` for constrained choices when possible.
  - Always pass a compact, serializable agent/world state; avoid raw objects.
- Logging & errors
  - Use `core.logger.setup_logger` and log at appropriate levels; prefer informative messages over prints.
  - Fail safe on model parsing: validate and sanitize outputs, apply fallbacks (e.g., schedule parsing tolerant to dashes/time noise).
- Exceptions
  - Guard LLM-dependent blocks with try/except and degrade gracefully.
  - Never let a failed generation block deterministic progression (executor/world should continue).
- Time handling
  - Normalize to minute resolution; prefer `replace(second=0, microsecond=0)` when computing end times.
- Imports
  - Keep imports explicit at module top; avoid implicit globals inside methods.

## 5. Common Patterns
- Template-driven LLM calls using Jinja2 + `LocalModel.generate`.
- Dual retrieval mode: vector relevance (pseudo-embeddings) with recency/importance weights, with keyword-index fallbacks.
- Address-based spatial targeting: compute sector/arena/object, with degradations and special markers.
- Deterministic executor
  - Dijkstra pathfinding with terrain costs (e.g., gravel preferred).
  - Event attachment/movement on tiles; idempotent interactions.
- Reflection loop
  - Focal points → insights with evidence → thoughts saved with embeddings/keywords; importance thresholds gate triggering.
- Planning pipeline
  - Wake time → day blocks → hourly schedule → select sector/arena/object → generate object interaction + microtasks.
  - Reaction to perceived events with constrained options (chat/observe/help/follow/wait/ignore).

## 6. Do's and Don'ts
- Do
  - Keep execution deterministic; isolate randomness and model outputs.
  - Validate and sanitize all model outputs; handle cross‑midnight times, malformed lines, and missing fields.
  - Use `allowed_strings` grammars to constrain model choices when a fixed set is known.
  - Keep memory small and relevant; avoid logging idle/no-op descriptions.
  - Use keyword indexing for fast retrieval and supplement with vector scoring.
  - Maintain separation of concerns: perception/retrieval/planner/executor/reflection.
  - Update address index (`TileManager`) via provided setters to keep cache fresh.
- Don’t
  - Don’t block the simulation loop on slow LLM calls; prefer timeboxing and graceful fallbacks.
  - Don’t store duplicate or trivial memories (e.g., idle) that bloat retrieval.
  - Don’t couple agent logic directly to tile internals; use `TileManager` APIs.
  - Don’t assume non-collidable tiles when computing addresses; always check or degrade.

## 7. Tools & Dependencies
- Key libraries
  - `llama-cpp-python` – Local LLM inference backend.
  - `jinja2` – Prompt templating.
  - `python-dotenv` – ENV configuration.
  - `transformers` – Present in requirements for model interop/examples; not required by the core runtime.
- ENV configuration (examples)
  - `MODEL_PATH` – Path to a GGUF model (required by `LocalModel`).
  - `N_CTX` – Context window tokens (default 2048).
  - `N_THREADS` – CPU threads used by llama.cpp (defaults to CPU cores).
  - `N_GPU_LAYERS` – Optional offload layers for Metal/CUDA builds.
  - Optional behavior tuning envs used in code: `CONTEXT_NODE_LIMIT`, `MAX_PROMPT_MEMORIES`.
- Setup (quick)
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - Export env vars above; ensure model path exists.
  - Run: `python main.py`

## 8. Other Notes
- LLM output constraints
  - Prefer short, well-formed outputs; keep to one-line choices when expected; avoid verbose prose in microtasks.
- Special markers and parsing
  - Time strings must be `HH:MM`; parser tolerates stray unicode dashes and minor noise but sanitize upstream where possible.
  - Address strings may include `<waiting> x y` to rendezvous near a target agent; executor will start chat upon arrival.
- Memory hygiene
  - Use `importance_trigger_current` to schedule reflection; avoid flooding with low-value entries.
- Extending the world
  - Use `WorldBuilder` helpers and `TileManager` setters to keep address cache consistent; define passable vs blocking objects.
- Determinism for tests
  - Inject a stubbed `LocalModel` into agents; seed any randomness; the pseudo-embedding is already deterministic.
