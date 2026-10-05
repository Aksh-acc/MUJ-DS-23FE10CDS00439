"""Typed access to config.yaml.

Config is loaded once and passed down explicitly rather than read from a global,
so tests can build a Config in memory without touching the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class BranchConfig:
    name: str
    label: str
    source: str
    description: str
    keywords: tuple[str, ...]
    documents: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    dense_weight: float
    sparse_weight: float

    @property
    def descriptor(self) -> str:
        """Text embedded to represent the branch during routing."""
        return f"{self.label}. {self.description.strip()} {' '.join(self.keywords)}"


@dataclass(frozen=True)
class Paths:
    corpus: Path
    chunks: Path
    artifacts: Path
    eval_queries: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    batch_size: int
    normalize: bool


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool
    model: str
    candidates: int


@dataclass(frozen=True)
class RouterConfig:
    strategy: str
    embedding_weight: float
    lexical_weight: float
    relative_threshold: float
    min_score: float
    max_branches: int
    always_keep_top: bool


@dataclass(frozen=True)
class RetrievalConfig:
    fusion: str
    rrf_k: int
    dense_candidates: int
    sparse_candidates: int


@dataclass(frozen=True)
class GenerationConfig:
    max_evidence_chars: int
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    groq_model: str
    groq_fallback_model: str
    ollama_model: str
    ollama_host: str
    local_hf_model: str


@dataclass(frozen=True)
class SentimentConfig:
    enabled: bool
    model: str
    max_samples: int


@dataclass(frozen=True)
class Config:
    paths: Paths
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    router: RouterConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    llm: LLMConfig
    sentiment: SentimentConfig
    branches: dict[str, BranchConfig] = field(default_factory=dict)

    @property
    def branch_names(self) -> list[str]:
        return list(self.branches)

    def branch(self, name: str) -> BranchConfig:
        try:
            return self.branches[name]
        except KeyError:
            raise KeyError(
                f"unknown branch {name!r}; configured branches: {self.branch_names}"
            ) from None


def from_dict(raw: dict[str, Any]) -> Config:
    branches = {
        name: BranchConfig(
            name=name,
            label=spec["label"],
            source=spec["source"],
            description=spec["description"],
            keywords=tuple(spec["keywords"]),
            documents=int(spec["documents"]),
            chunk_size=int(spec["chunk_size"]),
            chunk_overlap=int(spec["chunk_overlap"]),
            top_k=int(spec["top_k"]),
            dense_weight=float(spec["dense_weight"]),
            sparse_weight=float(spec["sparse_weight"]),
        )
        for name, spec in raw["branches"].items()
    }

    paths = raw["paths"]
    llm = raw["llm"]

    return Config(
        paths=Paths(
            corpus=_resolve(paths["corpus"]),
            chunks=_resolve(paths["chunks"]),
            artifacts=_resolve(paths["artifacts"]),
            eval_queries=_resolve(paths["eval_queries"]),
        ),
        embedding=EmbeddingConfig(**raw["embedding"]),
        reranker=RerankerConfig(**raw["reranker"]),
        router=RouterConfig(**raw["router"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        generation=GenerationConfig(**raw["generation"]),
        llm=LLMConfig(
            provider=llm["provider"],
            groq_model=llm["groq"]["model"],
            groq_fallback_model=llm["groq"].get("fallback_model", llm["groq"]["model"]),
            ollama_model=llm["ollama"]["model"],
            ollama_host=llm["ollama"]["host"],
            local_hf_model=llm["local_hf"]["model"],
        ),
        sentiment=SentimentConfig(**raw["sentiment"]),
        branches=branches,
    )


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return from_dict(yaml.safe_load(handle))
