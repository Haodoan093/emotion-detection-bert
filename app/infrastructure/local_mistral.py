from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.domain.ports import LLMService


class LocalMistralService(LLMService):
    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

    def _load_model(self) -> None:
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

    def generate(self, prompt: str) -> str:
        self._load_model()
        assert self._model is not None
        assert self._tokenizer is not None
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        output = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return text.strip()

