from fastapi import APIRouter, Depends, HTTPException, Request

from app.adapters.http.schemas import LlmRequest, LlmResponse
from app.application.rag_use_cases import GenerateRagAnswerUseCase

router = APIRouter(prefix="/llm", tags=["LLM"])


def get_use_case(request: Request) -> GenerateRagAnswerUseCase:
    use_case = getattr(request.app.state, "llm_use_case", None)
    if use_case is None:
        raise HTTPException(status_code=500, detail="LLM service is not configured.")
    return use_case


@router.post("/answer", response_model=LlmResponse)
def answer(
    payload: LlmRequest,
    use_case: GenerateRagAnswerUseCase = Depends(get_use_case),
) -> LlmResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    try:
        result = use_case.execute(question=question)
    except Exception as exc:
        message = str(exc)
        if "api key" in message.lower():
            raise HTTPException(
                status_code=500,
                detail="LLM API key is missing or invalid.",
            ) from exc
        raise HTTPException(
            status_code=500,
            detail="LLM request failed. Check server logs for details.",
        ) from exc
    return LlmResponse(answer=result.answer, citations=result.citations)
