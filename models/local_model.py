import pprint
from llama_cpp import Llama, LlamaGrammar
import os

from core.logger import setup_logger

logger = setup_logger(__name__)

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

        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=4,  # ToDo: make configureable: adjust to your CPU
            verbose=False,
        )

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
        # Ensure we don't exceed the model context window by
        # (1) clamping max_tokens and (2) truncating the prompt if needed.
        try:
            prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            # Reserve at least 1 token for generation
            safe_max_new = max(1, min(max_tokens, self.n_ctx - 1))
            if len(prompt_tokens) + safe_max_new > self.n_ctx:
                # Truncate prompt tokens to fit within context window.
                max_prompt_tokens = max(1, self.n_ctx - safe_max_new)
                # Keep both the beginning (instructions) and the tail (most recent details)
                if max_prompt_tokens <= 512:
                    head = max_prompt_tokens // 2
                else:
                    head = 256
                tail = max_prompt_tokens - head
                kept_tokens = prompt_tokens[:head] + prompt_tokens[-tail:]
                try:
                    truncated_prompt = self.llm.detokenize(kept_tokens).decode("utf-8", errors="ignore")
                except Exception:
                    # Fallback: rough char-based truncation if detokenize isn't available
                    approx_chars = max(2000, min(len(prompt), 4 * max_prompt_tokens))
                    truncated_prompt = prompt[: approx_chars // 2] + "\n...\n" + prompt[-(approx_chars // 2) :]
                logger.debug(
                    "Truncated prompt to fit context: orig_tokens=%d, new_tokens<=%d, max_new=%d",
                    len(prompt_tokens),
                    max_prompt_tokens,
                    safe_max_new,
                )
                prompt = truncated_prompt
            # Re-clamp max tokens in case prompt changed size significantly
            prompt_tokens2 = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            max_tokens = max(1, min(max_tokens, self.n_ctx - max(1, len(prompt_tokens2))))
        except Exception as e:
            # If anything goes wrong during token accounting, fall back to provided values
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
