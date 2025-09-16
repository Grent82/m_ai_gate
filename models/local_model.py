import pprint
from llama_cpp import Llama, LlamaGrammar
import os

from core.logger import setup_logger

logger = setup_logger(__name__, log_level="DEBUG")

class LocalModel:
    """Thin wrapper around :class:`llama_cpp.Llama`.

    The model path can be supplied directly or set via the ``MODEL_PATH``
    environment variable.

    Examples
    --------
    >>> LocalModel("/path/to/model.gguf")
    >>> LocalModel()  # expects MODEL_PATH to be defined
    """

    def __init__(self, model_path: str | None = None) -> None:
        model_path = model_path or os.environ.get("MODEL_PATH")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LLM model not found at: {model_path}. Set MODEL_PATH or pass a valid path."
            )

        # Allow overriding context window via env var (default 2048)
        self.n_ctx = int(os.environ.get("N_CTX", "2048"))
        # Allow tuning CPU threads and optional GPU offload via env
        # Defaults: use available CPU cores; no GPU offload unless specified
        self.n_threads = int(os.environ.get("N_THREADS", str(max(1, (os.cpu_count() or 4)))))
        n_gpu_layers_env = os.environ.get("N_GPU_LAYERS")

        llm_kwargs = dict(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )
        if n_gpu_layers_env is not None:
            try:
                llm_kwargs["n_gpu_layers"] = int(n_gpu_layers_env)
            except ValueError:
                # Ignore invalid values and stay on CPU
                pass

        self.llm = Llama(**llm_kwargs)

    def _choices_to_grammar(self, choices: list[str]) -> str:
        def esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace("\"", "\\\"")
        alts = " | ".join(f'"{esc(c)}"' for c in choices)
        return f"root ::= {alts}\n"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        stop: list | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        allowed_strings: list[str] | None = None,
    ) -> str:
        try:
            prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            safe_max_new = max(1, min(max_tokens, self.n_ctx - 1))
            if len(prompt_tokens) + safe_max_new > self.n_ctx:
                max_prompt_tokens = max(1, self.n_ctx - safe_max_new)
                head = 256 if max_prompt_tokens > 512 else max_prompt_tokens // 2
                tail = max_prompt_tokens - head
                kept_tokens = prompt_tokens[:head] + prompt_tokens[-tail:]
                try:
                    detok = self.llm.detokenize(kept_tokens).decode("utf-8", errors="ignore")
                except Exception:
                    approx_chars = max(2000, min(len(prompt), 4 * max_prompt_tokens))
                    detok = (
                        prompt[: approx_chars // 2]
                        + "\n...\n"
                        + prompt[-(approx_chars // 2) :]
                    )
                prompt = detok
            prompt_tokens2 = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            max_tokens = max(1, min(max_tokens, self.n_ctx - max(1, len(prompt_tokens2))))
        except Exception as e:
            logger.debug("Context management fallback due to: %s", e)
            max_tokens = max(1, min(max_tokens, self.n_ctx // 4))

        kwargs = dict(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=stop or ["</s>", "User:", "###"],
            temperature=temperature,
            top_p=top_p,
        )
        if allowed_strings:
            try:
                grammar_str = self._choices_to_grammar(allowed_strings)
                kwargs["grammar"] = LlamaGrammar.from_string(grammar_str)
            except Exception as e:
                logger.debug("Failed to build/apply grammar for choices %s: %s", allowed_strings, e)
        result = self.llm(**kwargs)
        logger.debug("Raw LLM output:\n%s", pprint.pformat(result))
        return result["choices"][0]["text"].strip()
