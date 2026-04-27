from pydantic import BaseModel, Field


class AsrResponse(BaseModel):
    text: str


class LlmRequest(BaseModel):
    question: str
    context: str | None = None


class Citation(BaseModel):
    source: str | None = None
    doc_id: str | None = None
    chunk_index: int | None = None


class LlmResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class EmotionPrediction(BaseModel):
    label: str
    score: float


class EmotionDetectRequest(BaseModel):
    text: str
    top_k: int = 3


class EmotionDetectResponse(BaseModel):
    text: str
    label: str
    score: float
    predictions: list[EmotionPrediction] = Field(default_factory=list)


class ChatEmotion(BaseModel):
    label: str
    score: float
    predictions: list[EmotionPrediction] = Field(default_factory=list)


class ChatVoiceResponse(BaseModel):
    text: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    audio_filename: str | None = None
    emotion: ChatEmotion | None = None
