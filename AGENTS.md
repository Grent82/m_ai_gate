# AGENTS.md

## Purpose

This repository contains an experimental Python-based simulation of autonomous in-world agents.
The folder `agents/` already contains domain agents and pipelines for the simulation itself.
This `AGENTS.md` defines a separate multi-agent setup for Codex-driven development work on the repository.

Use these development agents to analyze, extend, test, and stabilize the project without confusing them with the in-game agent logic.

The project direction is explicitly inspired by the paper *Generative Agents: Interactive Simulacra of Human Behavior* and the reference repository `joonspk-research/generative_agents`, but this repository should prioritize software quality, architecture, maintainability, testability, and extensibility over cloning the original implementation details.

## Project Analysis

### Current shape of the codebase

- Entry point: `main.py`
- Simulation core: `core/`
- In-world agent cognition modules: `agents/`
- Memory systems: `memory/`
- Local model integration: `models/`
- Interfaces and contracts: `interfaces/`
- Prompt templates: `prompts/`

### What matters technically

- The codebase is early-stage and research-like, not yet production-hardened.
- The target product direction is a cleaner, more maintainable implementation of a generative-agent architecture inspired by the original paper and reference repo.
- There is a strong prompt-driven architecture across planning, perception, reflection, and event extraction.
- The local LLM dependency is `llama-cpp-python`, configured via `MODEL_PATH`.
- There are already visible GitHub Actions quality gates under `.github/workflows/`:
  - `python-app.yml` runs `flake8` and `pytest`
  - `python-package.yml` runs `flake8` and `pytest`
- There is still no visible dedicated type-check configuration in the repository root.
- Test scaffolding is not yet visible as a conventional `tests/` directory, so runtime validation coverage should still be reviewed carefully.
- There are signs of encoding issues in several strings (`â€“`, `â€™`, `â€”`), so text quality and prompt hygiene need active maintenance.

### Main development needs

- Preserve the architectural intent of generative agents while avoiding a monolithic or tightly coupled implementation.
- Stabilize the runtime path for local model execution.
- Add tests around memory, world building, planner logic, and prompt parsing.
- Improve developer experience: run instructions, env validation, quality checks.
- Reduce regressions in prompt outputs and parser expectations.
- Keep architecture coherent while the simulation expands.

### Project positioning

- This repo is not a copy of the original `generative_agents` codebase.
- It is a reimplementation effort guided by the same broad architecture:
  - observation
  - memory
  - retrieval
  - reflection
  - planning
- Development choices should favor clean code and clear boundaries even when that differs from the structure of the reference implementation.

## Operating Model

Use one coordinator plus focused specialist agents.
The coordinator owns task decomposition, sequencing, acceptance criteria, and final integration.
Specialists own bounded areas of the repository and should avoid overlapping edits where possible.

Default rule:

- One agent plans.
- One agent owns architecture and refactors.
- One agent owns prompt quality and parser alignment.
- One agent owns tests and validation.
- One agent owns docs and onboarding.

Spawn fewer agents for small tasks.
Spawn more only when ownership is clear and write scopes do not collide.

## Development Agent Roster

### 1. `repo-orchestrator`

Role:
Primary coordinator for all development work in this repository.

Responsibilities:

- Read the task and map it onto the codebase.
- Decide which specialist agents are needed.
- Keep work split by directory or subsystem.
- Define acceptance criteria before parallel work starts.
- Review integration risks before merging edits.

Owns:

- Cross-cutting changes
- Final integration decisions
- Task sequencing
- Change summaries

When to use:

- Any non-trivial task
- Refactors crossing `core/`, `agents/`, `memory/`, or `models/`
- Work that mixes code, prompts, and docs

### 2. `simulation-architect`

Role:
Maintains structural coherence of the simulation and domain model.

Responsibilities:

- Refactor `core/`, `interfaces/`, and shared contracts.
- Keep `Agent`, `World`, tile, and action abstractions clean.
- Remove accidental coupling between runtime systems.
- Improve naming, boundaries, and data flow.

Owns:

- `core/`
- `interfaces/`
- Shared entity contracts

When to use:

- Architecture cleanup
- State model changes
- Runtime orchestration fixes

### 3. `cognition-engineer`

Role:
Owns the cognitive pipeline of the in-world agents.

Responsibilities:

- Improve planning, perception, retrieval, reflection, and event generation.
- Keep prompt outputs aligned with parser expectations.
- Reduce brittle LLM assumptions.
- Add fallback behavior when model outputs are malformed.

Owns:

- `agents/`
- Relevant pieces of `memory/`
- Prompt-to-parser glue

When to use:

- Planning bugs
- Reflection and retrieval issues
- Event-triple extraction problems

### 4. `prompt-operator`

Role:
Owns prompt assets as first-class system components.

Responsibilities:

- Rewrite templates for determinism and parseability.
- Standardize output formats across prompts.
- Add prompt fixtures or snapshot-style checks where practical.
- Detect encoding and formatting issues in prompt files.

Owns:

- `prompts/`
- Prompt conventions in parser code

When to use:

- LLM output drift
- Parsing failures
- Quality tuning for behavior prompts

### 5. `memory-systems-engineer`

Role:
Maintains short-term, long-term, and spatial memory behavior.

Responsibilities:

- Improve retrieval relevance and storage semantics.
- Protect invariants in memory node creation and access.
- Add tests for significance, retention, and recall behavior.
- Ensure memory APIs remain consistent across cognition modules.

Owns:

- `memory/`
- Retrieval interfaces touching memory

When to use:

- Memory bugs
- Retrieval quality issues
- Schema or persistence-style changes

### 6. `model-runtime-engineer`

Role:
Owns local model integration and execution ergonomics.

Responsibilities:

- Harden `LocalModel` setup and failure modes.
- Improve configuration of `MODEL_PATH`, threads, and context size.
- Make local execution easier to diagnose.
- Prepare the code for alternative backends if needed later.

Owns:

- `models/`
- Runtime bootstrap concerns in `main.py`

When to use:

- LLM startup failures
- Backend abstraction work
- Environment validation and runtime logging

### 7. `qa-and-tests`

Role:
Owns validation, regression checks, and reproducibility.

Responsibilities:

- Add unit tests and lightweight integration tests.
- Build fixtures for prompts, memories, and world setup.
- Create smoke tests that do not require a full local model when possible.
- Add validation for parser edge cases and encoding issues.

Owns:

- `tests/`
- Test scaffolding
- Quality commands and CI hooks

When to use:

- Any bug fix
- Any parser change
- Any architectural change

### 8. `devex-docs`

Role:
Owns onboarding, usage docs, and contributor workflow.

Responsibilities:

- Keep `README.md` accurate.
- Document environment setup and local model requirements.
- Add contributor guidance and task recipes.
- Document agent workflow and development conventions.

Owns:

- `README.md`
- Developer docs
- Supporting repo documentation

When to use:

- Setup friction
- Documentation drift
- New workflow introduction

## Recommended Skills

These are the useful Codex skill categories for this repository.
If local skills exist in your environment, map them to these responsibilities.

### Core skills to have

- `python-architecture`
  - Refactoring, module boundaries, dependency cleanup, interfaces
- `python-testing`
  - `pytest`, fixtures, mocks, parser regression tests
- `prompt-engineering`
  - Deterministic prompt design, output schemas, evaluation loops
- `llm-runtime`
  - Local model integration, config handling, fallback behavior
- `documentation`
  - README cleanup, contributor docs, architecture notes

### Nice-to-have skills

- `typing-and-static-analysis`
- `logging-and-observability`
- `performance-profiling`
- `text-encoding-repair`

## Recommended MCP / Tooling

### High-value MCPs

- `filesystem`
  - Read and edit repo files safely
- `git`
  - Inspect diffs, status, history, and branch state
- `github`
  - Useful if the repo is mirrored remotely and PR-based review is used
- `python`
  - Run tests, smoke checks, fixtures, and scripts

### Optional MCPs

- `docs` or `knowledge-base`
  - Useful for internal architecture notes or design references
- `issue-tracker`
  - Useful once work is tracked outside the repo

### Not required yet

- Browser automation
- Deployment MCPs
- Database MCPs

This repository is still local-development-centric, so keep the toolchain small.

## Plugins

No custom Codex plugin is strictly required yet.
Add plugins only when they remove repeated friction.

### Plugins that would actually help

- `pytest runner`
  - Single-command test execution and summary
- `quality gate`
  - Run lint, tests, and basic type checks together
- `prompt snapshot checker`
  - Compare prompt outputs or parser expectations against fixtures
- `env doctor`
  - Validate `MODEL_PATH`, Python version, and optional runtime dependencies

### Plugins that are premature right now

- Deployment plugins
- Release automation plugins
- Complex multi-repo orchestration plugins

## Suggested Task Routing

### For bug fixes

- `repo-orchestrator`
- `qa-and-tests`
- Relevant domain specialist

### For prompt failures

- `prompt-operator`
- `cognition-engineer`
- `qa-and-tests`

### For architecture work

- `simulation-architect`
- `memory-systems-engineer`
- `qa-and-tests`

### For setup and onboarding

- `devex-docs`
- `model-runtime-engineer`

## Coordination Rules

- Never confuse development agents with the simulation modules in `agents/`.
- Prefer disjoint ownership by directory.
- Do not let multiple agents edit the same prompt file or parser path without explicit coordination.
- Every behavioral change should include at least one validation step.
- Changes to prompt format must be mirrored by parser or test updates.
- Keep `README.md` aligned with the actual entry point and workflow.
- Prefer smoke tests that do not require a heavy local model for every run.

## Minimal Workflow Standard

For non-trivial work, the coordinator should do this:

1. Inspect impacted files.
2. State assumptions and acceptance criteria.
3. Delegate by subsystem.
4. Integrate changes.
5. Run validation.
6. Update docs if developer workflow changed.

## Immediate Next Improvements For This Repo

These should be the first tasks for the new development agent system:

1. Add a `tests/` directory with smoke tests for world building, planner parsing, and memory primitives.
2. Add a lightweight quality command set, ideally `pytest` first.
3. Introduce environment validation for `MODEL_PATH`.
4. Normalize prompt text encoding and parser assumptions.
5. Continue separating domain logic from infrastructure and runtime concerns.

## Directory Convention For Future Agent Assets

If you later want repo-local prompt files or agent-specific working notes for Codex, use this layout:

```text
.agents/
  roles/
    repo-orchestrator.md
    simulation-architect.md
    cognition-engineer.md
    prompt-operator.md
    memory-systems-engineer.md
    model-runtime-engineer.md
    qa-and-tests.md
    devex-docs.md
  workflows/
    bugfix.md
    prompt-change.md
    refactor.md
```

Do not place these files in the existing `agents/` directory because that directory already belongs to the application domain.

## Definition of Done

A task is only complete when:

- the responsible subsystem was changed coherently,
- validation was run or explicitly documented as blocked,
- docs were updated if workflow or setup changed,
- and integration risk across prompts, parsers, and memory flow was checked.
