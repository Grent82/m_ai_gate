import pprint
from llama_cpp import Llama, LlamaGrammar
import os
import re

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
        """
        Generate text. If the prompt contains a '### System' section, render
        it using the Llama 3.1 Instruct chat template (manual formatting),
        otherwise treat it as a plain completion prompt.
        """

        def _extract_system_and_user_blocks(txt: str) -> tuple[str | None, str]:
            # Pull out the System block; everything else becomes User content.
            sys_pat = re.compile(r"^###\s*System:?\s*$([\s\S]*?)(?=^###\s*\w|\Z)", re.MULTILINE)
            m = sys_pat.search(txt)
            system = m.group(1).strip() if m else None
            txt_wo_system = txt[: m.start()] + txt[m.end():] if m else txt
            # Remove any trailing '### Assistant' block if present
            asst_pat = re.compile(r"^###\s*Assistant:?\s*$", re.MULTILINE)
            asst_m = asst_pat.search(txt_wo_system)
            user = txt_wo_system[: asst_m.start()].strip() if asst_m else txt_wo_system.strip()
            return system, user

        def _as_llama31_chat(system: str | None, user: str) -> str:
            sys_part = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}\n<|eot_id|>"
                if system else "<|begin_of_text|>"
            )
            return (
                f"{sys_part}"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}\n<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        # Build stop list and ensure <|eot_id|> for clean assistant turn termination
        stop_list = list(stop or [])
        if "<|eot_id|>" not in stop_list:
            stop_list.append("<|eot_id|>")

        # If this is a structured template with System section, convert to chat
        if re.search(r"^###\s*System", prompt, re.MULTILINE):
            system, user = _extract_system_and_user_blocks(prompt)
            chat_str = _as_llama31_chat(system, user)

            # Context accounting for chat string
            try:
                toks = self.llm.tokenize(chat_str.encode("utf-8"), add_bos=True)
                safe_max_new = max(1, min(max_tokens, self.n_ctx - 1))
                if len(toks) + safe_max_new > self.n_ctx:
                    max_prompt_tokens = max(1, self.n_ctx - safe_max_new)
                    head = 256 if max_prompt_tokens > 512 else max_prompt_tokens // 2
                    tail = max_prompt_tokens - head
                    kept = toks[:head] + toks[-tail:]
                    try:
                        chat_str = self.llm.detokenize(kept).decode("utf-8", errors="ignore")
                    except Exception:
                        approx_chars = max(2000, min(len(chat_str), 4 * max_prompt_tokens))
                        chat_str = chat_str[: approx_chars // 2] + "\n...\n" + chat_str[-(approx_chars // 2) :]
                toks2 = self.llm.tokenize(chat_str.encode("utf-8"), add_bos=True)
                max_tokens = max(1, min(max_tokens, self.n_ctx - max(1, len(toks2))))
            except Exception as e:
                logger.debug("Chat context accounting failed: %s", e)
                max_tokens = max(1, min(max_tokens, self.n_ctx // 4))

            kwargs = dict(
                prompt=chat_str,
                max_tokens=max_tokens,
                stop=stop_list,
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
            logger.debug("Raw LLM output (chat):\n%s", pprint.pformat(result))
            return result["choices"][0]["text"].strip()

        # Plain completion path
        try:
            prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            safe_max_new = max(1, min(max_tokens, self.n_ctx - 1))
            if len(prompt_tokens) + safe_max_new > self.n_ctx:
                max_prompt_tokens = max(1, self.n_ctx - safe_max_new)
                head = 256 if max_prompt_tokens > 512 else max_prompt_tokens // 2
                tail = max_prompt_tokens - head
                kept_tokens = prompt_tokens[:head] + prompt_tokens[-tail:]
                try:
                    prompt = self.llm.detokenize(kept_tokens).decode("utf-8", errors="ignore")
                except Exception:
                    approx_chars = max(2000, min(len(prompt), 4 * max_prompt_tokens))
                    prompt = prompt[: approx_chars // 2] + "\n...\n" + prompt[-(approx_chars // 2) :]
            prompt_tokens2 = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            max_tokens = max(1, min(max_tokens, self.n_ctx - max(1, len(prompt_tokens2))))
        except Exception as e:
            logger.debug("Context management fallback due to: %s", e)
            max_tokens = max(1, min(max_tokens, self.n_ctx // 4))

        kwargs = dict(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=stop_list or ["</s>", "User:", "###"],
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
