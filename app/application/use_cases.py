from app.domain.entities import AsrResult, ChatVoiceResult, EmotionResult, LlmResult
from app.application.rag_use_cases import GenerateRagAnswerUseCase
from app.domain.ports import ASRService, EmotionService, LLMService
from app.utils.audio import cleanup_temp_audio, write_temp_audio


class TranscribeAudioUseCase:
    def __init__(self, asr_service: ASRService) -> None:
        self._asr_service = asr_service

    def execute(self, audio_bytes: bytes, filename: str | None) -> AsrResult:
        audio_path = write_temp_audio(audio_bytes, filename)
        try:
            text = self._asr_service.transcribe(audio_path)
        finally:
            cleanup_temp_audio(audio_path)
        return AsrResult(text=text)


class GenerateFinancialAnswerUseCase:
    def __init__(self, llm_service: LLMService) -> None:
        self._llm_service = llm_service

    def execute(self, question: str, context: str | None) -> LlmResult:
        prompt = self._build_prompt(question=question, context=context)
        answer = self._llm_service.generate(prompt)
        return LlmResult(answer=answer)

    @staticmethod
    def _build_prompt(question: str, context: str | None) -> str:
        context_block = context.strip() if context else ""
        return (
            "Ban la chuyen gia tai chinh. "
            "Chi tra loi dua tren context duoc cung cap. "
            "Neu context khong du thong tin, hay noi ro.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Cau hoi:\n{question}\n\n"
            "Tra loi ngan gon, ro rang, dung context."
        )


class DetectEmotionUseCase:
    def __init__(self, emotion_service: EmotionService) -> None:
        self._emotion_service = emotion_service

    def execute(self, text: str, top_k: int = 3) -> EmotionResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Text is required.")
        normalized_top_k = max(1, top_k)
        return self._emotion_service.detect(normalized_text, top_k=normalized_top_k)


class ChatVoiceUseCase:
    def __init__(
        self,
        transcribe_use_case: TranscribeAudioUseCase,
        rag_use_case: GenerateRagAnswerUseCase,
        emotion_use_case: DetectEmotionUseCase | None = None,
    ) -> None:
        self._transcribe_use_case = transcribe_use_case
        self._rag_use_case = rag_use_case
        self._emotion_use_case = emotion_use_case

    def execute(
        self,
        audio_bytes: bytes | None,
        filename: str | None,
        question: str | None = None,
    ) -> ChatVoiceResult:
        text = question.strip() if question else ""
        if not text:
            if not audio_bytes:
                raise ValueError("Either audio or question is required.")
            asr_result = self._transcribe_use_case.execute(audio_bytes, filename)
            text = asr_result.text

        emotion_res: EmotionResult | None = None
        if self._emotion_use_case and text:
            try:
                emotion_res = self._emotion_use_case.execute(text)
            except Exception:
                pass

        emotion_label = emotion_res.label if emotion_res else None
        emotion_score = emotion_res.score if emotion_res else 1.0
        rag_result = self._rag_use_case.execute(
            question=text, 
            emotion=emotion_label,
            emotion_score=emotion_score
        )
        return ChatVoiceResult(
            text=text,
            answer=rag_result.answer,
            citations=rag_result.citations,
            emotion=emotion_res,
        )
