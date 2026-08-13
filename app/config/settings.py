"""Application settings loaded from environment variables.

Uses Pydantic Settings (v2) so configuration is typed, validated, and
loaded once from `.env` / process environment — similar to Spring's
`@ConfigurationProperties` with `application.yml`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]


class Settings(BaseSettings):
    """Central application configuration.

    Attributes:
        app_name: Human-readable service name used in logs and health checks.
        app_env: Deployment environment (development, staging, production).
        app_debug: Enables verbose behaviour suitable for local development.
        log_level: Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        host: Bind address for the ASGI server.
        port: Bind port for the ASGI server.
        api_prefix: URL prefix for versioned API routes.
        supported_document_types: Comma-separated file extensions allowed for ingestion.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        separators: Ordered split separators for recursive character splitting.
        embedding_model: Sentence-Transformers / Hugging Face model id.
        embed_batch_size: Texts per inference batch.
        normalize_embeddings: Whether to L2-normalize output vectors.
        vector_db: Vector database backend key (currently ``chroma``).
        chroma_collection_name: Chroma collection to create / use.
        chroma_persist_directory: Local path for Chroma persistence files.
        top_k: Default number of neighbors returned by retrieval.
        default_score_threshold: Default minimum cosine similarity ``[0, 1]``.
        openai_api_key: API key for the OpenAI provider (blank allowed until call).
        openai_model: Chat model id for OpenAI generation.
        temperature: Sampling temperature in ``[0, 2]``.
        max_tokens: Maximum completion tokens (must be ``> 0``).
        openai_timeout_seconds: Optional HTTP timeout for OpenAI calls.
        max_context_tokens: Soft threshold for prompt token estimates (warn only).
        prompt_template_directory: Directory containing RAG prompt templates.
        rag_default_top_k: Default retrieval depth for RAG orchestration.
        rag_default_score_threshold: Default similarity floor for RAG queries.
        enable_evaluation: Master switch for the evaluation framework.
        default_pass_threshold: Overall report pass cut-off in ``[0, 1]``.
        metric_timeout: Per-metric wall-clock timeout in seconds (``> 0``).
        langsmith_api_key: API key for LangSmith tracing (blank allowed when disabled).
        langsmith_project: LangSmith project name.
        enable_langsmith: Master switch for LangSmith tracing.
        enable_deepeval: Register DeepEval-backed metrics when building registries.
        enable_tool_validation: Run tool-validation step in automation runners.
        report_directory: Directory for evaluation JSON/CSV/HTML reports.
        benchmark_directory: Directory for benchmark summary artifacts.
        golden_dataset_path: Default path to the golden evaluation dataset.
        input_token_cost_per_1k: USD cost per 1K prompt tokens (estimates).
        output_token_cost_per_1k: USD cost per 1K completion tokens (estimates).
        quality_gate_enabled: Master switch for Sprint 12 quality gates.
        min_faithfulness: Minimum faithfulness score gate.
        max_hallucination: Maximum hallucination score gate.
        min_relevancy: Minimum answer relevancy gate.
        min_context_precision: Minimum context precision gate.
        min_context_recall: Minimum context recall gate.
        max_tool_failures: Maximum tool failures allowed.
        max_tool_latency: Maximum tool latency in ms.
        max_total_latency: Maximum end-to-end latency in ms.
        max_cost: Maximum estimated USD cost.
        quality_pass_threshold: Overall score PASS cut-off (PASS_THRESHOLD).
        warning_threshold: Overall score WARNING band floor.
        benchmark_history_path: Path to benchmark history JSON.
        quality_report_directory: Directory for quality_report artifacts.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="rag-evaluation-platform", description="Service name")
    app_env: str = Field(default="development", description="Runtime environment")
    app_debug: bool = Field(default=False, description="Debug mode flag")
    log_level: str = Field(default="INFO", description="Logging level")
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port")
    api_prefix: str = Field(default="/api/v1", description="API route prefix")
    supported_document_types: str = Field(
        default=".pdf",
        description="Comma-separated list of allowed document extensions",
    )

    chunk_size: int = Field(default=1000, description="Target max characters per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between consecutive chunks")
    separators: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_SEPARATORS),
        description="Ordered separators for RecursiveCharacterTextSplitter",
    )

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence-Transformers / Hugging Face model id",
    )
    embed_batch_size: int = Field(
        default=32,
        description="Number of texts encoded per inference batch",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="L2-normalize vectors (preferred for cosine similarity)",
    )

    vector_db: str = Field(
        default="chroma",
        description="Vector database backend identifier (chroma or json)",
    )
    chroma_collection_name: str = Field(
        default="rag_documents",
        description="Chroma collection name for persisted embeddings",
    )
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        description="Local directory for Chroma persistence files",
    )

    top_k: int = Field(
        default=5,
        description="Default number of nearest neighbors to retrieve",
    )
    default_score_threshold: float = Field(
        default=0.0,
        description="Default minimum cosine similarity score in [0, 1]",
    )

    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (validated at provider call time if blank)",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model identifier",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM sampling temperature in [0, 2]",
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum completion tokens for generation",
    )
    openai_timeout_seconds: float | None = Field(
        default=60.0,
        description="Optional OpenAI request timeout in seconds",
    )

    max_context_tokens: int = Field(
        default=8000,
        description="Soft max estimated prompt tokens before warning (no truncation)",
    )
    prompt_template_directory: str = Field(
        default="prompts/templates",
        description="Directory of RAG prompt templates (relative to app/ or cwd)",
    )
    rag_default_top_k: int = Field(
        default=5,
        description="Default top-k for RAG orchestration when request omits it",
    )
    rag_default_score_threshold: float = Field(
        default=0.0,
        description="Default score threshold for RAG orchestration when omitted",
    )

    enable_evaluation: bool = Field(
        default=True,
        description="Master switch for the evaluation framework",
    )
    default_pass_threshold: float = Field(
        default=0.7,
        description="Overall EvaluationReport pass threshold in [0, 1]",
    )
    metric_timeout: float = Field(
        default=30.0,
        description="Per-metric evaluation timeout in seconds",
    )

    langsmith_api_key: str = Field(
        default="",
        description="LangSmith API key (optional when ENABLE_LANGSMITH=false)",
    )
    langsmith_project: str = Field(
        default="rag-evaluation",
        description="LangSmith project name for traces",
    )
    enable_langsmith: bool = Field(
        default=False,
        description="Master switch for LangSmith tracing",
    )
    enable_deepeval: bool = Field(
        default=True,
        description="Register DeepEval adapters in default metric registries",
    )
    enable_tool_validation: bool = Field(
        default=True,
        description="Run tool validation during evaluation automation",
    )
    report_directory: str = Field(
        default="./data/reports",
        description="Output directory for evaluation reports",
    )
    benchmark_directory: str = Field(
        default="./data/benchmarks",
        description="Output directory for benchmark summaries",
    )
    golden_dataset_path: str = Field(
        default="./datasets/golden_dataset.json",
        description="Default golden dataset JSON path",
    )
    input_token_cost_per_1k: float = Field(
        default=0.00015,
        description="Estimated USD per 1K prompt tokens",
    )
    output_token_cost_per_1k: float = Field(
        default=0.0006,
        description="Estimated USD per 1K completion tokens",
    )

    # Sprint 12 — Continuous AI Evaluation / Quality Gates
    quality_gate_enabled: bool = Field(
        default=True,
        description="Master switch for quality-gate evaluation",
    )
    min_faithfulness: float = Field(
        default=0.7,
        description="Minimum faithfulness score for quality gates",
    )
    max_hallucination: float = Field(
        default=0.3,
        description="Maximum hallucination score for quality gates",
    )
    min_relevancy: float = Field(
        default=0.7,
        description="Minimum answer relevancy for quality gates",
    )
    min_context_precision: float = Field(
        default=0.6,
        description="Minimum context precision for quality gates",
    )
    min_context_recall: float = Field(
        default=0.6,
        description="Minimum context recall for quality gates",
    )
    max_tool_failures: int = Field(
        default=0,
        description="Maximum allowed tool failures per evaluation",
    )
    max_tool_latency: float = Field(
        default=60_000.0,
        description="Maximum tool latency in milliseconds",
    )
    max_total_latency: float = Field(
        default=120_000.0,
        description="Maximum end-to-end latency in milliseconds",
    )
    max_cost: float = Field(
        default=1.0,
        description="Maximum estimated USD cost per evaluation",
    )
    quality_pass_threshold: float = Field(
        default=0.7,
        description="Overall score PASS threshold (PASS_THRESHOLD env alias)",
        validation_alias=AliasChoices(
            "PASS_THRESHOLD",
            "quality_pass_threshold",
            "QUALITY_PASS_THRESHOLD",
        ),
    )
    warning_threshold: float = Field(
        default=0.6,
        description="Overall score WARNING band floor",
    )
    benchmark_history_path: str = Field(
        default="./data/benchmarks/history.json",
        description="JSON file storing benchmark history runs",
    )
    quality_report_directory: str = Field(
        default="./data/quality_reports",
        description="Output directory for quality_report.* artifacts",
    )
    employee_data_path: str = Field(
        default="./data/employees/employees.json",
        description="JSON path for the deterministic structured employee dataset",
    )

    # Sprint 15 — MCP tool interoperability (Direct mode remains the safe default)
    buddie_tool_mode: str = Field(
        default="direct",
        description="Tool execution mode: direct (local ToolRegistry) or mcp",
        validation_alias=AliasChoices(
            "BUDDIE_TOOL_MODE",
            "buddie_tool_mode",
            "TOOL_MODE",
        ),
    )
    mcp_transport: str = Field(
        default="memory",
        description="MCP transport: memory | stdio | http",
        validation_alias=AliasChoices(
            "MCP_TRANSPORT",
            "mcp_transport",
        ),
    )
    mcp_server_url: str = Field(
        default="",
        description=(
            "MCP Streamable HTTP URL (used when mcp_transport=http). "
            "Example: http://mcp-server:8100/mcp — host comes from deployment config."
        ),
        validation_alias=AliasChoices(
            "MCP_SERVER_URL",
            "mcp_server_url",
        ),
    )
    mcp_server_command: str = Field(
        default="python -m app.mcp",
        description="Shell command for stdio MCP server (space-separated)",
        validation_alias=AliasChoices(
            "MCP_SERVER_COMMAND",
            "mcp_server_command",
        ),
    )
    mcp_timeout_seconds: float = Field(
        default=30.0,
        description="Per-tool MCP client timeout in seconds",
        validation_alias=AliasChoices(
            "MCP_TIMEOUT_SECONDS",
            "mcp_timeout_seconds",
        ),
    )

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_must_be_positive(cls, value: int) -> int:
        """Reject non-positive chunk sizes."""
        if value <= 0:
            raise ValueError("CHUNK_SIZE must be a positive integer")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def chunk_overlap_must_be_non_negative(cls, value: int) -> int:
        """Reject negative overlap values."""
        if value < 0:
            raise ValueError("CHUNK_OVERLAP must be >= 0")
        return value

    @field_validator("separators")
    @classmethod
    def separators_must_be_non_empty(cls, value: list[str]) -> list[str]:
        """Require at least one separator entry (empty string is allowed as last resort)."""
        if not value:
            raise ValueError("SEPARATORS must be a non-empty list")
        return value

    @field_validator("embedding_model")
    @classmethod
    def embedding_model_must_be_non_empty(cls, value: str) -> str:
        """Reject blank embedding model ids."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("EMBEDDING_MODEL must be a non-empty string")
        return cleaned

    @field_validator("embed_batch_size")
    @classmethod
    def embed_batch_size_must_be_positive(cls, value: int) -> int:
        """Reject non-positive batch sizes."""
        if value <= 0:
            raise ValueError("EMBED_BATCH_SIZE must be a positive integer")
        return value

    @field_validator("vector_db")
    @classmethod
    def vector_db_must_be_supported(cls, value: str) -> str:
        """Accept only known vector database backends."""
        cleaned = value.strip().lower()
        if cleaned not in {"chroma", "json"}:
            raise ValueError(
                f"VECTOR_DB must be one of: chroma, json (got '{value}')"
            )
        return cleaned

    @field_validator("chroma_collection_name")
    @classmethod
    def chroma_collection_name_must_be_valid(cls, value: str) -> str:
        """Reject blank or too-short Chroma collection names."""
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError(
                "CHROMA_COLLECTION_NAME must be at least 3 characters"
            )
        return cleaned

    @field_validator("chroma_persist_directory")
    @classmethod
    def chroma_persist_directory_must_be_non_empty(cls, value: str) -> str:
        """Reject blank persistence paths."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("CHROMA_PERSIST_DIRECTORY must be a non-empty path")
        return cleaned

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_positive(cls, value: int) -> int:
        """Reject non-positive retrieval depths."""
        if value <= 0:
            raise ValueError("TOP_K must be a positive integer")
        return value

    @field_validator("default_score_threshold")
    @classmethod
    def default_score_threshold_must_be_unit_interval(cls, value: float) -> float:
        """Require cosine-normalized thresholds in ``[0, 1]``."""
        if value < 0.0 or value > 1.0:
            raise ValueError(
                "DEFAULT_SCORE_THRESHOLD must be between 0 and 1 inclusive"
            )
        return value

    @field_validator("openai_model")
    @classmethod
    def openai_model_must_be_non_empty(cls, value: str) -> str:
        """Reject blank OpenAI model ids."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("OPENAI_MODEL must be a non-empty string")
        return cleaned

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_in_range(cls, value: float) -> float:
        """Require temperature in the OpenAI-supported ``[0, 2]`` range."""
        if value < 0.0 or value > 2.0:
            raise ValueError("TEMPERATURE must be between 0 and 2 inclusive")
        return value

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_must_be_positive(cls, value: int) -> int:
        """Reject non-positive max token budgets."""
        if value <= 0:
            raise ValueError("MAX_TOKENS must be a positive integer")
        return value

    @field_validator("openai_timeout_seconds")
    @classmethod
    def openai_timeout_must_be_positive_when_set(
        cls,
        value: float | None,
    ) -> float | None:
        """Allow ``None`` (SDK default) or a positive timeout in seconds."""
        if value is None:
            return value
        if value <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be > 0 when set")
        return value

    @field_validator("max_context_tokens")
    @classmethod
    def max_context_tokens_must_be_positive(cls, value: int) -> int:
        """Reject non-positive context token thresholds."""
        if value <= 0:
            raise ValueError("MAX_CONTEXT_TOKENS must be a positive integer")
        return value

    @field_validator("prompt_template_directory")
    @classmethod
    def prompt_template_directory_must_be_non_empty(cls, value: str) -> str:
        """Reject blank prompt template directories."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("PROMPT_TEMPLATE_DIRECTORY must be a non-empty path")
        return cleaned

    @field_validator("rag_default_top_k")
    @classmethod
    def rag_default_top_k_must_be_positive(cls, value: int) -> int:
        """Reject non-positive RAG default top-k values."""
        if value <= 0:
            raise ValueError("RAG_DEFAULT_TOP_K must be a positive integer")
        return value

    @field_validator("rag_default_score_threshold")
    @classmethod
    def rag_default_score_threshold_must_be_unit_interval(cls, value: float) -> float:
        """Require RAG default score thresholds in ``[0, 1]``."""
        if value < 0.0 or value > 1.0:
            raise ValueError(
                "RAG_DEFAULT_SCORE_THRESHOLD must be between 0 and 1 inclusive"
            )
        return value

    @field_validator("default_pass_threshold")
    @classmethod
    def default_pass_threshold_must_be_unit_interval(cls, value: float) -> float:
        """Require evaluation pass thresholds in ``[0, 1]``."""
        if value < 0.0 or value > 1.0:
            raise ValueError(
                "DEFAULT_PASS_THRESHOLD must be between 0 and 1 inclusive"
            )
        return value

    @field_validator("metric_timeout")
    @classmethod
    def metric_timeout_must_be_positive(cls, value: float) -> float:
        """Reject non-positive metric timeouts."""
        if value <= 0:
            raise ValueError("METRIC_TIMEOUT must be > 0")
        return value

    @field_validator("langsmith_project")
    @classmethod
    def langsmith_project_must_be_non_empty(cls, value: str) -> str:
        """Reject blank LangSmith project names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("LANGSMITH_PROJECT must be a non-empty string")
        return cleaned

    @field_validator("input_token_cost_per_1k", "output_token_cost_per_1k")
    @classmethod
    def token_costs_must_be_non_negative(cls, value: float) -> float:
        """Reject negative token cost estimates."""
        if value < 0.0:
            raise ValueError("Token cost settings must be >= 0")
        return value

    @field_validator(
        "min_faithfulness",
        "max_hallucination",
        "min_relevancy",
        "min_context_precision",
        "min_context_recall",
        "quality_pass_threshold",
        "warning_threshold",
    )
    @classmethod
    def quality_unit_interval_settings(cls, value: float) -> float:
        """Require quality score thresholds in ``[0, 1]``."""
        if value < 0.0 or value > 1.0:
            raise ValueError("Quality score thresholds must be between 0 and 1")
        return value

    @field_validator("max_tool_failures")
    @classmethod
    def max_tool_failures_must_be_non_negative(cls, value: int) -> int:
        """Reject negative tool failure budgets."""
        if value < 0:
            raise ValueError("MAX_TOOL_FAILURES must be >= 0")
        return value

    @field_validator("max_tool_latency", "max_total_latency", "max_cost")
    @classmethod
    def quality_budget_must_be_non_negative(cls, value: float) -> float:
        """Reject negative latency / cost budgets."""
        if value < 0.0:
            raise ValueError("Quality latency/cost budgets must be >= 0")
        return value

    @field_validator("buddie_tool_mode")
    @classmethod
    def buddie_tool_mode_must_be_supported(cls, value: str) -> str:
        """Accept only direct or mcp tool execution modes."""
        cleaned = value.strip().lower()
        if cleaned not in {"direct", "mcp"}:
            raise ValueError("BUDDIE_TOOL_MODE must be one of: direct, mcp")
        return cleaned

    @field_validator("mcp_transport")
    @classmethod
    def mcp_transport_must_be_supported(cls, value: str) -> str:
        """Accept memory, stdio, or http MCP transports."""
        cleaned = value.strip().lower()
        if cleaned not in {"memory", "stdio", "http"}:
            raise ValueError("MCP_TRANSPORT must be one of: memory, stdio, http")
        return cleaned

    @field_validator("mcp_timeout_seconds")
    @classmethod
    def mcp_timeout_must_be_positive(cls, value: float) -> float:
        """Reject non-positive MCP timeouts."""
        if value <= 0:
            raise ValueError("MCP_TIMEOUT_SECONDS must be > 0")
        return value

    @field_validator(
        "report_directory",
        "benchmark_directory",
        "golden_dataset_path",
        "benchmark_history_path",
        "quality_report_directory",
    )
    @classmethod
    def path_settings_must_be_non_empty(cls, value: str) -> str:
        """Reject blank filesystem path settings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Path settings must be non-empty strings")
        return cleaned

    @model_validator(mode="after")
    def overlap_cannot_exceed_chunk_size(self) -> Settings:
        """Ensure overlap is not larger than the chunk window."""
        if self.chunk_overlap > self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) cannot exceed "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        if self.warning_threshold > self.quality_pass_threshold:
            raise ValueError(
                "WARNING_THRESHOLD cannot exceed PASS_THRESHOLD "
                f"({self.warning_threshold} > {self.quality_pass_threshold})"
            )
        return self

    def get_supported_extensions(self) -> frozenset[str]:
        """Parse ``supported_document_types`` into a normalized extension set.

        Extensions are lowercased and ensured to start with a leading dot
        (e.g. ``pdf`` and ``.pdf`` both become ``.pdf``).

        Returns:
            Immutable set of allowed file extensions.
        """
        extensions: set[str] = set()
        for part in self.supported_document_types.split(","):
            cleaned = part.strip().lower()
            if not cleaned:
                continue
            if not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            extensions.add(cleaned)
        return frozenset(extensions)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    Caching avoids re-reading the environment on every request. Clear the
    cache in tests with ``get_settings.cache_clear()`` when overriding env vars.

    Returns:
        Validated application settings instance.
    """
    return Settings()
