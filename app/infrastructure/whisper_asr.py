import warnings

import torch
import whisper

from app.domain.ports import ASRService


class WhisperASRService(ASRService):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_fp16: bool | None = None,
        language: str | None = "vi",
        initial_prompt: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._use_fp16 = self._resolve_fp16(use_fp16)
        self._language = language.strip().lower() if language and language.strip() else None
        self._initial_prompt = initial_prompt.strip() if initial_prompt and initial_prompt.strip() else None
        self._model = None

    def _resolve_device(self, requested_device: str) -> str:
        device = requested_device.strip().lower()
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA was requested but is not available; falling back to CPU.")
            return "cpu"
        return device

    def _resolve_fp16(self, requested_fp16: bool | None) -> bool:
        if requested_fp16 is None:
            return self._device.startswith("cuda")
        if requested_fp16 and not self._device.startswith("cuda"):
            warnings.warn("FP16 requested without CUDA device; using FP32.")
            return False
        return requested_fp16

    def load_model(self) -> None:
        if self._model is None:
            self._model = whisper.load_model(self._model_name, device=self._device)

    def transcribe(self, audio_path: str) -> str:
        self.load_model()
        transcribe_kwargs = {
            "fp16": self._use_fp16,
            "task": "transcribe",
        }
        if self._language:
            transcribe_kwargs["language"] = self._language
        if self._initial_prompt:
            transcribe_kwargs["initial_prompt"] = self._initial_prompt
        result = self._model.transcribe(audio_path, **transcribe_kwargs)
        text = result.get("text", "")
        return text.strip()

