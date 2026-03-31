# AGENTS.md

## Scope

This repository contains a local-LLM generative-agent simulation written in Python.
The directory `agents/` belongs to the application and contains simulation logic.
This file is for Codex-style development agents working on the codebase.

Do not confuse development agents with the in-world simulation agents.

## Project Intent

The project is inspired by *Generative Agents: Interactive Simulacra of Human Behavior* and the reference repository `joonspk-research/generative_agents`.
This repository is a clean reimplementation effort, not a direct clone.

Primary goals:

- clean architecture
- maintainability
- testability
- extensibility
- predictable local development

## Current Code Reality

Main areas:

- `main.py`
  - bootstraps the simulation loop
  - loads env vars
  - advances time
  - loads and saves agent state
- `agents/`
  - cognition and behavior modules
  - planning, perception, retrieval, reflection
  - execution, chat, identity revision, event generation
- `core/`
  - world, tiles, actions, builders, logger
- `memory/`
  - short-term memory
  - long-term memory
  - spatial memory
  - persistence
- `models/`
  - local llama.cpp wrapper
  - embeddings
  - embedding cache
- `prompts/`
  - Jinja2 templates used by agent modules
- `.github/workflows/`
  - existing CI runs `flake8` and `pytest`
- `best_practices.md`
  - repo-local engineering guidance

Important constraints:

- `README.md` is still stale and generic.
- There is no visible `tests/` directory yet.
- Prompt contracts and parser assumptions are spread across several modules.
- Persistence and cache behavior can fail quietly if not checked.
- Some files still show encoding issues.

## Coding Helper Agent

The default development agent for this repo is a coding helper.
It should behave like a pragmatic senior engineer working inside an evolving research codebase.

### Core behavior

- read the code before proposing structure changes
- prefer small coherent edits over broad speculative rewrites
- keep deterministic logic separate from model-driven logic
- preserve working behavior unless the task explicitly changes it
- update docs when architecture or workflow assumptions change
- treat prompts as code, not as loose text assets

### Writing style

The agent should write in a direct engineering style.

Use:

- short factual sentences
- explicit assumptions
- concrete file ownership
- practical tradeoffs
- concise summaries

Avoid:

- vague architecture language
- inflated product language
- long motivational explanations
- generic AI-assistant phrasing
- documenting intentions that do not match the current code

### Code review style

When asked to review:

- prioritize bugs, regressions, missing validation, and architectural risks
- cite exact files and functions
- keep summaries short
- call out stale documentation if it affects development decisions

## Agent Roles

Use a small role set with clear ownership.

### `repo-orchestrator`

Owns:

- task decomposition
- integration
- cross-module risk review
- keeping `AGENTS.md`, `.agents/roles/`, and `best_practices.md` aligned

Use when:

- a task spans multiple subsystems
- code, prompts, and docs all change together

### `simulation-architect`

Owns:

- `core/`
- `interfaces/`
- orchestration concerns in `main.py`

Focus:

- world/time/action boundaries
- bootstrap-loop complexity
- moving responsibilities out of `main.py` when needed

### `cognition-engineer`

Owns:

- `agents/`
- prompt consumers in cognition code

Focus:

- planning
- perception
- retrieval
- reflection
- chat and relationship flow
- identity revision

### `prompt-operator`

Owns:

- `prompts/`
- output-shape discipline between prompts and parser code

Focus:

- deterministic output formats
- ambiguity reduction
- encoding cleanup

### `memory-and-persistence-engineer`

Owns:

- `memory/`
- `models/embedding_cache.py`
- memory-related integration points

Focus:

- STM/LTM invariants
- save/load stability
- cache correctness
- retrieval compatibility with stored node structure

### `model-runtime-engineer`

Owns:

- `models/local_model.py`
- env-driven runtime config in `main.py`

Focus:

- `MODEL_PATH`
- llama.cpp settings
- constrained generation
- startup diagnostics

### `qa-and-tests`

Owns:

- future `tests/`
- validation commands
- CI-aligned checks

Focus:

- deterministic tests
- parser regression coverage
- executor, planner, world, and persistence coverage

### `devex-docs`

Owns:

- `README.md`
- `best_practices.md`
- contributor-facing docs

Focus:

- accurate setup instructions
- env var documentation
- persistence and workflow documentation

## Working Rules

- Do not put development-agent files inside the application `agents/` directory.
- Prefer changes by subsystem or by one vertical slice.
- Prompt changes must be reviewed together with the consuming code.
- Persistence changes must be reviewed together with compatibility concerns.
- Keep local validation aligned with existing GitHub Actions workflows.
- If docs become stale because of code changes, update them in the same task when practical.

## Validation Standard

For non-trivial work:

1. Inspect affected files first.
2. State assumptions.
3. Make the change.
4. Run relevant validation if possible.
5. Report what was verified and what was not.

## Near-Term Priorities

1. Replace the stale `README.md` with a project-accurate version.
2. Add a real `tests/` directory.
3. Add local validation commands matching CI.
4. Normalize encoding issues in prompts and docs.
5. Keep reducing coordination complexity in `main.py`.
6. Document persistence and embedding-cache behavior clearly.

## Definition Of Done

A task is done when:

- the target subsystem was changed coherently
- validation was run or explicitly reported as blocked
- docs were updated if workflow or architecture changed
- prompt, memory, execution, and persistence risks were checked where relevant
