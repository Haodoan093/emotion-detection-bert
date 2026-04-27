import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.http.routes import router as asr_router
from app.adapters.http.llm_routes import router as llm_router
from app.adapters.http.chat_routes import router as chat_router
from app.adapters.http.emotion_routes import router as emotion_router
from app.application.rag_use_cases import GenerateRagAnswerUseCase, QuerySimilarChunksUseCase
from app.application.use_cases import ChatVoiceUseCase, DetectEmotionUseCase, TranscribeAudioUseCase
from app.core.config import settings
from app.infrastructure.hf_emotion import HfEmotionService
from app.infrastructure.langchain_retriever import SupabaseRetriever
from app.infrastructure.sentence_transformer_embeddings import SentenceTransformerEmbeddingService
from app.infrastructure.supabase_vector_store import SupabaseVectorStore
from app.infrastructure.whisper_asr import WhisperASRService

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    docs_url=settings.docs_url,
    openapi_url=settings.openapi_url,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _configure_rag(transcribe_use_case: TranscribeAudioUseCase) -> None:
    app.state.llm_use_case = None
    app.state.chat_voice_use_case = None

    if not settings.rag_enabled:
        logger.info("RAG startup is disabled via RAG_ENABLED.")
        return
    if not settings.supabase_url or not settings.supabase_service_key:
        logger.warning("RAG disabled: SUPABASE_URL and SUPABASE_SERVICE_KEY are not set.")
        return

    llm_provider = settings.llm_provider.lower()
    if llm_provider not in ["gemini", "openai"]:
        logger.warning("RAG disabled: unsupported LLM_PROVIDER '%s'.", settings.llm_provider)
        return
    if llm_provider == "gemini" and not settings.gemini_api_key:
        logger.warning("RAG disabled: GEMINI_API_KEY is missing.")
        return
    if llm_provider == "openai" and not settings.openai_api_key:
        logger.warning("RAG disabled: OPENAI_API_KEY is missing.")
        return

    try:
        from langchain_classic.chains import RetrievalQA
        from langchain_core.prompts import PromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        logger.warning("RAG disabled: missing optional dependency (%s).", exc)
        return

    try:
        embedding_service = SentenceTransformerEmbeddingService(
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
        )
        vector_store = SupabaseVectorStore(
            url=settings.supabase_url,
            key=settings.supabase_service_key,
            table=settings.supabase_table,
            match_rpc=settings.supabase_match_rpc,
        )
        query_use_case = QuerySimilarChunksUseCase(embedding_service, vector_store)
        retriever = SupabaseRetriever(query_use_case, top_k=settings.rag_top_k)

        if llm_provider == "openai":
            llm = ChatOpenAI(
                model=settings.openai_model_name,
                api_key=settings.openai_api_key,
                temperature=0,
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0,
            )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Bạn là một chuyên gia tư vấn tài chính ảo với 10 năm kinh nghiệm, có chứng chỉ hành nghề tại Việt Nam. "
                "Nhiệm vụ của bạn là giải đáp thắc mắc tài chính dựa trên tài liệu được cung cấp. "
                "Thông tin cảm xúc của khách hàng đã được nhúng vào câu hỏi đầu vào.\n\n"
                "[TÀI LIỆU THAM KHẢO - BẮT BUỘC DỰA VÀO ĐÂY]\n"
                "{context}\n\n"
                "[QUY TẮC CHẤT LƯỢNG - KHÔNG ĐƯỢC VI PHẠM]\n"
                "1. ✅ CHỈ sử dụng thông tin trong tài liệu trên. TUYỆT ĐỐI KHÔNG bịa số liệu, lãi suất, quy định.\n"
                "2. ✅ Nếu tài liệu KHÔNG có thông tin -> thành thật nói: 'Tôi cần kiểm tra thêm thông tin này từ chuyên viên.'\n"
                "3. ✅ Khi cite số liệu -> ghi rõ nguồn: 'theo quy định [tên văn bản/điều khoản]'\n"
                "4. ✅ Tiếng Việt chuẩn mực, chuyên nghiệp, tránh Hán-Việt phức tạp.\n"
                "5. ✅ Độ dài: Ngắn gọn nhưng đủ ý (3-5 câu cho phần tư vấn chính).\n\n"
                "[CÂU HỎI CỦA KHÁCH HÀNG]\n"
                "{question}\n\n"
                "[CẤU TRÚC PHẢN HỒI BẮT BUỘC]\n"
                "1) 🤝 Đồng cảm mở đầu (1 câu): Thể hiện hiểu cảm xúc khách, dùng tone và câu mở đầu gợi ý.\n"
                "2) 📋 Nội dung tư vấn chính (2-3 câu): Dựa HOÀN TOÀN trên tài liệu, có cite nguồn.\n"
                "3) 💡 Lời khuyên / Cảnh báo / Follow-up (1 câu): Gợi ý hành động tiếp theo hoặc câu hỏi mở.\n\n"
                "Phản hồi của bạn:"
            ),
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        rag_use_case = GenerateRagAnswerUseCase(qa_chain)
        app.state.llm_use_case = rag_use_case
        app.state.chat_voice_use_case = ChatVoiceUseCase(
            transcribe_use_case, 
            rag_use_case,
            getattr(app.state, "emotion_use_case", None)
        )
    except Exception:
        logger.exception("RAG initialization failed; continuing without RAG.")


@app.on_event("startup")
def startup() -> None:
    asr_service = WhisperASRService(
        model_name=settings.asr_model_name,
        device=settings.asr_device,
        use_fp16=settings.asr_use_fp16,
        language=settings.asr_language,
        initial_prompt=settings.asr_initial_prompt,
    )
    asr_service.load_model()
    transcribe_use_case = TranscribeAudioUseCase(asr_service)
    app.state.transcribe_use_case = transcribe_use_case

    app.state.emotion_use_case = None
    if settings.emotion_enabled:
        emotion_service = HfEmotionService(
            model_name=settings.emotion_model_name,
            device=settings.emotion_device,
        )
        emotion_service.load_model()
        app.state.emotion_use_case = DetectEmotionUseCase(emotion_service)

    _configure_rag(transcribe_use_case)


@app.get("/")
def root() -> dict:
    return {"message": "ASR API is running"}


app.include_router(asr_router)
app.include_router(llm_router)
app.include_router(chat_router)
app.include_router(emotion_router)
