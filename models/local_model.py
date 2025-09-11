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

        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
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
