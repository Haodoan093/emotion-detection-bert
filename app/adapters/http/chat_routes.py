import base64
import io
import json
import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app.adapters.http.schemas import ChatEmotion, ChatVoiceResponse, Citation, EmotionPrediction
from app.application.use_cases import ChatVoiceUseCase, DetectEmotionUseCase
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

PARTIAL_MIN_BYTES = 32_000
PARTIAL_EVERY_CHUNKS = 4

EVENT_ALIASES = {
    "start": "start",
    "begin": "start",
    "open": "start",
    "end": "end",
    "stop": "end",
    "close": "end",
    "ping": "ping",
    "chunk": "chunk",
    "audio": "chunk",
    "media": "chunk",
}


def get_use_case(request: Request) -> ChatVoiceUseCase:
    use_case = getattr(request.app.state, "chat_voice_use_case", None)
    if use_case is None:
        raise HTTPException(status_code=500, detail="Chat voice service is not configured.")
    return use_case


def _format_internal_error(exc: Exception, generic_message: str) -> HTTPException:
    message = str(exc)
    if "ffmpeg" in message.lower():
        return HTTPException(
            status_code=500,
            detail="Transcription failed. ffmpeg is required to read m4a files.",
        )
    return HTTPException(status_code=500, detail=generic_message)


def _normalize_event(payload: dict) -> str:
    raw_event = payload.get("event") or payload.get("type") or payload.get("action") or payload.get("command")
    event = str(raw_event or "").strip().lower()
    return EVENT_ALIASES.get(event, event)


def _extract_chunk_bytes(payload: dict) -> bytes | None:
    for key in ("audio", "chunk", "data", "payload"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return base64.b64decode(value)
            except Exception:
                return None
    return None


def _build_tts_payload(answer_text: str) -> tuple[str | None, str | None, str | None]:
    text = answer_text.strip()
    if not text:
        return None, None, None

    try:
        from gtts import gTTS
    except Exception:
        return None, None, None

    language = settings.asr_language or "vi"
    try:
        buffer = io.BytesIO()
        gTTS(text=text, lang=language).write_to_fp(buffer)
    except Exception:
        return None, None, None

    audio_base64 = base64.b64encode(cast(Any, buffer.getvalue())).decode("ascii")
    return audio_base64, "audio/mpeg", "voice-response.mp3"


async def _detect_emotion_payload(app_state: Any, text: str) -> ChatEmotion | None:
    emotion_use_case = cast(DetectEmotionUseCase | None, getattr(app_state, "emotion_use_case", None))
    if emotion_use_case is None:
        return None
    try:
        logger.info(f"Detecting emotion for text: {text[:50]}...")
        emotion_result = await run_in_threadpool(emotion_use_case.execute, text, 3)
        logger.info(f"Detected emotion: {emotion_result.label}")
    except Exception as exc:
        logger.error(f"Emotion detection failed: {exc}")
        return None
    return ChatEmotion(
        label=emotion_result.label,
        score=emotion_result.score,
        predictions=[
            EmotionPrediction(label=item.label, score=item.score) for item in emotion_result.predictions
        ],
    )


@router.post("/voice", response_model=ChatVoiceResponse)
async def voice_chat(
    request: Request,
    question: str | None = Form(None),
    file: UploadFile | None = File(None),
    use_case: ChatVoiceUseCase = Depends(get_use_case),
) -> ChatVoiceResponse:
    audio_bytes: bytes | None = None
    filename: str | None = None
    if file is not None:
        audio_bytes = await file.read()
        filename = file.filename
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")
    if not question and audio_bytes is None:
        raise HTTPException(status_code=400, detail="Provide audio file or question text.")
    try:
        result = use_case.execute(
            audio_bytes=audio_bytes,
            filename=filename,
            question=question,
        )
    except Exception as exc:
        raise _format_internal_error(exc, "Voice chat failed. Check server logs for details.") from exc
    audio_base64, audio_mime_type, audio_filename = _build_tts_payload(result.answer)
    
    emotion_payload: ChatEmotion | None = None
    if result.emotion:
        emotion_payload = ChatEmotion(
            label=result.emotion.label,
            score=result.emotion.score,
            predictions=[
                EmotionPrediction(label=item.label, score=item.score) 
                for item in result.emotion.predictions
            ],
        )

    return ChatVoiceResponse(
        text=result.text,
        answer=result.answer,
        citations=[
            Citation(
                source=citation.source,
                doc_id=citation.doc_id,
                chunk_index=citation.chunk_index,
            )
            for citation in result.citations
        ],
        audio_base64=audio_base64,
        audio_mime_type=audio_mime_type,
        audio_filename=audio_filename,
        emotion=emotion_payload,
    )


@router.websocket("/voice/ws")
async def voice_chat_stream(websocket: WebSocket) -> None:
    chat_use_case = getattr(websocket.app.state, "chat_voice_use_case", None)
    transcribe_use_case = getattr(websocket.app.state, "transcribe_use_case", None)
    if chat_use_case is None or transcribe_use_case is None:
        await websocket.close(code=1011, reason="Voice chat service is not configured.")
        return

    await websocket.accept()
    await websocket.send_json(
        {
            "event": "ready",
            "message": "Send start/end as JSON and audio chunks as binary WebSocket frames.",
        }
    )

    audio_buffer = bytearray()
    filename: str | None = "stream.wav"
    question: str | None = None
    is_started = False
    chunk_counter = 0
    last_partial_text = ""

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                return
            if message_type != "websocket.receive":
                continue

            payload_bytes = message.get("bytes")
            if payload_bytes is not None:
                if not is_started:
                    await websocket.send_json({"event": "error", "detail": "Send start event before audio chunks."})
                    continue
                if not payload_bytes:
                    continue

                audio_buffer.extend(payload_bytes)
                chunk_counter += 1

                if len(audio_buffer) >= PARTIAL_MIN_BYTES and chunk_counter >= PARTIAL_EVERY_CHUNKS:
                    chunk_counter = 0
                    try:
                        partial_result = await run_in_threadpool(
                            transcribe_use_case.execute,
                            bytes(audio_buffer),
                            filename,
                        )
                    except Exception as exc:
                        await websocket.send_json(
                            {
                                "event": "error",
                                "detail": _format_internal_error(
                                    exc,
                                    "Streaming ASR failed. Check server logs for details.",
                                ).detail,
                            }
                        )
                        continue

                    partial_text = partial_result.text.strip()
                    if partial_text and partial_text != last_partial_text:
                        last_partial_text = partial_text
                        await websocket.send_json({"event": "partial", "text": partial_text})
                continue

            payload_text = message.get("text")
            if payload_text is None:
                continue

            payload_text = payload_text.strip()
            if payload_text.lower() in EVENT_ALIASES:
                payload = {"event": payload_text.lower()}
            else:
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    await websocket.send_json({"event": "error", "detail": "Invalid JSON message."})
                    continue

            event = _normalize_event(payload)
            if event == "start":
                is_started = True
                audio_buffer.clear()
                chunk_counter = 0
                last_partial_text = ""
                filename = payload.get("filename") or "stream.wav"
                raw_question = payload.get("question")
                question = raw_question.strip() if isinstance(raw_question, str) else None
                await websocket.send_json({"event": "started"})
                continue

            if event == "chunk":
                if not is_started:
                    await websocket.send_json({"event": "error", "detail": "Send start event before audio chunks."})
                    continue
                payload_chunk = _extract_chunk_bytes(payload)
                if payload_chunk is None:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "detail": "Invalid chunk payload. Send binary frames or base64 in audio/chunk/data/payload.",
                        }
                    )
                    continue
                if payload_chunk:
                    audio_buffer.extend(payload_chunk)
                    chunk_counter += 1
                continue

            if event == "end":
                if not is_started:
                    await websocket.send_json({"event": "error", "detail": "Session is not started."})
                    continue
                if not question and not audio_buffer:
                    await websocket.send_json(
                        {"event": "error", "detail": "Provide audio chunks or question text before end event."}
                    )
                    continue

                try:
                    result = await run_in_threadpool(
                        chat_use_case.execute,
                        bytes(audio_buffer) if audio_buffer else None,
                        filename,
                        question,
                    )
                except Exception as exc:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "detail": _format_internal_error(
                                exc,
                                "Voice chat failed. Check server logs for details.",
                            ).detail,
                        }
                    )
                    continue

                audio_base64, audio_mime_type, audio_filename = _build_tts_payload(result.answer)
                
                emotion_payload = None
                if result.emotion:
                    emotion_payload = {
                        "label": result.emotion.label,
                        "score": result.emotion.score,
                        "predictions": [
                            {"label": item.label, "score": item.score} for item in result.emotion.predictions
                        ],
                    }

                await websocket.send_json(
                    {
                        "event": "final",
                        "text": result.text,
                        "answer": result.answer,
                        "citations": [
                            {
                                "source": citation.source,
                                "doc_id": citation.doc_id,
                                "chunk_index": citation.chunk_index,
                            }
                            for citation in result.citations
                        ],
                        "audio_base64": audio_base64,
                        "audio_mime_type": audio_mime_type,
                        "audio_filename": audio_filename,
                        "emotion": emotion_payload,
                    }
                )
                await websocket.close(code=1000)
                return

            if event == "ping":
                await websocket.send_json({"event": "pong"})
                continue

            await websocket.send_json(
                {
                    "event": "error",
                    "detail": (
                        "Unsupported event. Use start/end JSON events, plain 'start'/'end' text, "
                        "or send audio via binary frames / base64 chunk events."
                    ),
                }
            )
    except WebSocketDisconnect:
        return


