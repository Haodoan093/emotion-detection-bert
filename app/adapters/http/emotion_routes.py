from fastapi import APIRouter, Depends, HTTPException, Request

from app.adapters.http.schemas import EmotionDetectRequest, EmotionDetectResponse, EmotionPrediction
from app.application.use_cases import DetectEmotionUseCase

router = APIRouter(prefix="/emotion", tags=["Emotion"])


def get_use_case(request: Request) -> DetectEmotionUseCase:
    use_case = getattr(request.app.state, "emotion_use_case", None)
    if use_case is None:
        raise HTTPException(status_code=503, detail="Emotion service is not configured.")
    return use_case


@router.post("/detect", response_model=EmotionDetectResponse)
def detect_emotion(
    payload: EmotionDetectRequest,
    use_case: DetectEmotionUseCase = Depends(get_use_case),
) -> EmotionDetectResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    try:
        result = use_case.execute(text=payload.text, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Emotion detection failed. Check server logs for details.",
        ) from exc

    return EmotionDetectResponse(
        text=result.text,
        label=result.label,
        score=result.score,
        predictions=[EmotionPrediction(label=item.label, score=item.score) for item in result.predictions],
    )

