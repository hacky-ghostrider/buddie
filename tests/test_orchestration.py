"""Tests for the RAG orchestration layer (Sprint 8).



Retriever, PromptBuilder, and LLMProvider are fully mocked — unit tests

must never call OpenAI or Chroma.

"""



from __future__ import annotations



from unittest.mock import MagicMock



import pytest

from fastapi.testclient import TestClient

from pydantic import ValidationError



from app.api.deps import get_rag_service

from app.config.settings import Settings

from app.generation.exceptions import (

    EmptyQuestionError,

    GenerationTimeoutError,

    PromptTemplateError,

)

from app.generation.models import BuiltPrompt, GeneratedAnswer

from app.main import create_app

from app.orchestration.exceptions import (

    EmptyRAGQuestionError,

    LLMOrchestrationError,

    PromptOrchestrationError,

    RetrievalOrchestrationError,

)

from app.orchestration.models import RAGRequest, RAGResponse

from app.orchestration.rag_service import RAGService

from app.retrieval.exceptions import RetrievalSearchError

from app.retrieval.models import RetrievedDocument





def _doc(

    text: str = "Recursive chunking splits on separators.",

    *,

    doc_id: str = "c1",

    score: float = 0.91,

) -> RetrievedDocument:

    return RetrievedDocument(

        id=doc_id,

        text=text,

        metadata={"file_name": "guide.pdf"},

        score=score,

    )





def _prompt(question: str = "What is chunking?") -> BuiltPrompt:

    return BuiltPrompt(

        system="You are a grounded assistant.",

        user=f"Context:\nchunk\n\nQuestion:\n{question}",

        question=question,

        context_document_count=1,

        context_char_length=20,

    )





def _answer(text: str = "Chunking splits long text.") -> GeneratedAnswer:

    return GeneratedAnswer.from_parts(

        answer=text,

        model="gpt-4o-mini",

        prompt_tokens=40,

        completion_tokens=10,

        finish_reason="stop",

    )





@pytest.fixture

def orchestration_settings() -> Settings:

    """Settings with a low context threshold for overflow-warning tests."""

    return Settings(

        app_env="test",

        rag_default_top_k=3,

        rag_default_score_threshold=0.1,

        max_context_tokens=10_000,

        prompt_template_directory="prompts/templates",

    )





@pytest.fixture

def mock_retriever() -> MagicMock:

    retriever = MagicMock()

    retriever.retrieve.return_value = [_doc()]

    return retriever





@pytest.fixture

def mock_prompt_builder() -> MagicMock:

    builder = MagicMock()

    builder.build.return_value = _prompt()

    builder.estimate_token_count.return_value = 50

    return builder





@pytest.fixture

def mock_llm() -> MagicMock:

    provider = MagicMock()

    provider.generate.return_value = _answer()

    return provider





@pytest.fixture

def service(

    mock_retriever: MagicMock,

    mock_prompt_builder: MagicMock,

    mock_llm: MagicMock,

    orchestration_settings: Settings,

) -> RAGService:

    return RAGService(

        retriever=mock_retriever,

        prompt_builder=mock_prompt_builder,

        llm_provider=mock_llm,

        settings=orchestration_settings,

    )





# ---------------------------------------------------------------------------

# Models

# ---------------------------------------------------------------------------





class TestRAGModels:

    def test_request_rejects_blank_top_k_override(self) -> None:

        with pytest.raises(ValidationError, match="top_k"):

            RAGRequest(question="Q?", top_k=0)



    def test_request_rejects_invalid_score_threshold(self) -> None:

        with pytest.raises(ValidationError, match="score_threshold"):

            RAGRequest(question="Q?", score_threshold=1.5)





# ---------------------------------------------------------------------------

# RAGService

# ---------------------------------------------------------------------------





class TestRAGService:

    def test_happy_path(

        self,

        service: RAGService,

        mock_retriever: MagicMock,

        mock_prompt_builder: MagicMock,

        mock_llm: MagicMock,

    ) -> None:

        response = service.query(RAGRequest(question="What is chunking?"))



        assert isinstance(response, RAGResponse)

        assert response.question == "What is chunking?"

        assert response.answer == "Chunking splits long text."

        assert len(response.retrieved_documents) == 1

        assert response.retrieved_documents[0].id == "c1"

        assert response.correlation_id

        assert response.retrieval_metadata["retrieved_count"] == 1

        assert response.retrieval_metadata["top_k"] == 3

        assert response.generation_metadata["model"] == "gpt-4o-mini"

        assert response.generation_metadata["prompt_tokens"] == 40

        assert response.generation_metadata["estimated_prompt_tokens"] == 50



        mock_retriever.retrieve.assert_called_once()

        mock_prompt_builder.build.assert_called_once()

        mock_llm.generate.assert_called_once()



    def test_empty_question(self, service: RAGService, mock_retriever: MagicMock) -> None:

        with pytest.raises(EmptyRAGQuestionError):

            service.query(RAGRequest(question="   "))

        mock_retriever.retrieve.assert_not_called()



    def test_no_retrieved_documents(

        self,

        service: RAGService,

        mock_retriever: MagicMock,

        mock_prompt_builder: MagicMock,

    ) -> None:

        mock_retriever.retrieve.return_value = []

        mock_prompt_builder.build.return_value = BuiltPrompt(

            system="sys",

            user="Context:\nnone\n\nQuestion:\nQ?",

            question="Q?",

            context_document_count=0,

            context_char_length=4,

        )



        response = service.query(RAGRequest(question="Q?"))



        assert response.retrieved_documents == []

        assert response.retrieval_metadata["retrieved_count"] == 0

        assert response.answer



    def test_retriever_failure(

        self,

        service: RAGService,

        mock_retriever: MagicMock,

    ) -> None:

        mock_retriever.retrieve.side_effect = RetrievalSearchError("chroma down")

        with pytest.raises(RetrievalOrchestrationError, match="Retrieval failed"):

            service.query(RAGRequest(question="Q?"))



    def test_prompt_failure(

        self,

        service: RAGService,

        mock_prompt_builder: MagicMock,

    ) -> None:

        mock_prompt_builder.build.side_effect = PromptTemplateError("missing template")

        with pytest.raises(PromptOrchestrationError, match="Prompt build failed"):

            service.query(RAGRequest(question="Q?"))



    def test_prompt_empty_question_mapped(

        self,

        service: RAGService,

        mock_prompt_builder: MagicMock,

    ) -> None:

        mock_prompt_builder.build.side_effect = EmptyQuestionError("blank")

        with pytest.raises(EmptyRAGQuestionError):

            service.query(RAGRequest(question="Q?"))



    def test_llm_failure(self, service: RAGService, mock_llm: MagicMock) -> None:

        mock_llm.generate.side_effect = GenerationTimeoutError("timed out")

        with pytest.raises(LLMOrchestrationError, match="timed out"):

            service.query(RAGRequest(question="Q?"))



    def test_latency_populated(self, service: RAGService) -> None:

        response = service.query(RAGRequest(question="What is chunking?"))



        assert response.latency.retrieval_ms >= 0.0

        assert response.latency.prompt_build_ms >= 0.0

        assert response.latency.llm_ms >= 0.0

        assert response.latency.total_ms >= 0.0

        assert response.generation_metadata["retrieval_latency_ms"] == (

            response.latency.retrieval_ms

        )

        assert response.generation_metadata["prompt_build_latency_ms"] == (

            response.latency.prompt_build_ms

        )

        assert response.generation_metadata["llm_latency_ms"] == response.latency.llm_ms

        assert response.generation_metadata["total_latency_ms"] == (

            response.latency.total_ms

        )



    def test_response_mapping_uses_request_overrides(

        self,

        service: RAGService,

        mock_retriever: MagicMock,

    ) -> None:

        filters = {"file_name": "guide.pdf"}

        response = service.query(

            RAGRequest(

                question="What is chunking?",

                top_k=2,

                score_threshold=0.5,

                metadata_filters=filters,

            )

        )



        mock_retriever.retrieve.assert_called_once_with(

            "What is chunking?",

            top_k=2,

            score_threshold=0.5,

            metadata_filters=filters,

        )

        assert response.retrieval_metadata["top_k"] == 2

        assert response.retrieval_metadata["score_threshold"] == 0.5

        assert response.retrieval_metadata["metadata_filters"] == filters



    def test_context_overflow_warns_but_continues(

        self,

        mock_retriever: MagicMock,

        mock_prompt_builder: MagicMock,

        mock_llm: MagicMock,

        caplog: pytest.LogCaptureFixture,

    ) -> None:

        settings = Settings(

            app_env="test",

            max_context_tokens=10,

            rag_default_top_k=3,

            prompt_template_directory="prompts/templates",

        )

        mock_prompt_builder.estimate_token_count.return_value = 500

        svc = RAGService(

            retriever=mock_retriever,

            prompt_builder=mock_prompt_builder,

            llm_provider=mock_llm,

            settings=settings,

        )



        with caplog.at_level("WARNING"):

            response = svc.query(RAGRequest(question="Q?"))



        assert response.generation_metadata["context_exceeded_threshold"] is True

        assert any("exceeds configured context" in msg for msg in caplog.messages)

        mock_llm.generate.assert_called_once()





# ---------------------------------------------------------------------------

# Settings

# ---------------------------------------------------------------------------





class TestOrchestrationSettings:

    def test_max_context_tokens_must_be_positive(self) -> None:

        with pytest.raises(ValidationError, match="MAX_CONTEXT_TOKENS"):

            Settings(max_context_tokens=0)



    def test_rag_defaults_validated(self) -> None:

        with pytest.raises(ValidationError, match="RAG_DEFAULT_TOP_K"):

            Settings(rag_default_top_k=0)

        with pytest.raises(ValidationError, match="RAG_DEFAULT_SCORE_THRESHOLD"):

            Settings(rag_default_score_threshold=1.5)



    def test_prompt_template_directory_required(self) -> None:

        with pytest.raises(ValidationError, match="PROMPT_TEMPLATE_DIRECTORY"):

            Settings(prompt_template_directory="  ")





# ---------------------------------------------------------------------------

# API mapping

# ---------------------------------------------------------------------------





class TestRAGAPI:

    def test_query_endpoint_maps_response(self, service: RAGService) -> None:

        application = create_app(

            settings=Settings(app_env="test", app_debug=True)

        )

        application.dependency_overrides[get_rag_service] = lambda: service

        client = TestClient(application)



        payload = {"question": "What is chunking?", "top_k": 2}

        result = client.post("/api/v1/rag/query", json=payload)



        assert result.status_code == 200

        body = result.json()

        assert body["question"] == "What is chunking?"

        assert body["answer"] == "Chunking splits long text."

        assert body["latency"]["total_ms"] >= 0.0

        assert "correlation_id" in body

        assert body["retrieval_metadata"]["top_k"] == 2



        application.dependency_overrides.clear()



    def test_query_endpoint_empty_question(self, service: RAGService) -> None:

        application = create_app(settings=Settings(app_env="test"))

        application.dependency_overrides[get_rag_service] = lambda: service

        client = TestClient(application)



        result = client.post("/api/v1/rag/query", json={"question": "  "})

        assert result.status_code == 400



        application.dependency_overrides.clear()



    def test_query_endpoint_retriever_failure(

        self,

        service: RAGService,

        mock_retriever: MagicMock,

    ) -> None:

        mock_retriever.retrieve.side_effect = RetrievalSearchError("down")

        application = create_app(settings=Settings(app_env="test"))

        application.dependency_overrides[get_rag_service] = lambda: service

        client = TestClient(application)



        result = client.post("/api/v1/rag/query", json={"question": "Q?"})

        assert result.status_code == 502



        application.dependency_overrides.clear()


