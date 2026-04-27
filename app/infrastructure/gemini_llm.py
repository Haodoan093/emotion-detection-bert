from google import generativeai

from app.domain.ports import LLMService


class GeminiLLMService(LLMService):
    def __init__(self, api_key: str, model_name: str) -> None:
        generativeai.configure(api_key=api_key)
        self._model = generativeai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        text = getattr(response, "text", "")
        return text.strip()

