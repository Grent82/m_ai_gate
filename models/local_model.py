import pprint
from llama_cpp import Llama
import os

from core.logger import setup_logger

logger = setup_logger(__name__)

class LocalModel:
    def __init__(self, model_path: str = "/Users/adittrich/privat/mistral-7b-instruct-v0.1.Q4_K_M.gguf"): # ToDo: make configureable
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model not found at: {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,    # ToDo: make configureable: adjust to your CPU
            verbose=False
        )

    def generate(self, prompt: str, max_tokens: int = 256, stop: list = None) -> str:
        result = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=stop or ["</s>", "User:", "###"]
        )
        logger.debug("Raw LLM output:\n%s", pprint.pformat(result))
        return result["choices"][0]["text"].strip()
