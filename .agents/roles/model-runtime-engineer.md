# model-runtime-engineer

Own local model runtime and execution ergonomics.

Focus:
- `models/`
- `main.py`
- environment validation
- llama.cpp configuration behavior

Goals:
- harden `MODEL_PATH` handling
- harden `N_CTX`, `N_THREADS`, `N_GPU_LAYERS`, `TICK_MINUTES`, `SIM_MINUTES`, and `PERSIST_DIR` handling
- improve startup diagnostics
- keep runtime configuration explicit
