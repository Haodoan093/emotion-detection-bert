from fastapi import APIRouter, Depends, File, Request, UploadFile, HTTPException

from app.adapters.http.schemas import AsrResponse
from app.application.use_cases import TranscribeAudioUseCase

router = APIRouter(prefix="/asr", tags=["ASR"])


def get_use_case(request: Request) -> TranscribeAudioUseCase:
    return request.app.state.transcribe_use_case


@router.post("/transcribe", response_model=AsrResponse)
async def transcribe(
    file: UploadFile = File(...),
    use_case: TranscribeAudioUseCase = Depends(get_use_case),
) -> AsrResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        result = use_case.execute(audio_bytes=audio_bytes, filename=file.filename)
    except Exception as exc:
        message = str(exc)
        if "ffmpeg" in message.lower():
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. ffmpeg is required to read m4a files.",
            ) from exc
        raise HTTPException(
            status_code=500,
            detail="Transcription failed. Check server logs for details.",
        ) from exc
    return AsrResponse(text=result.text)
