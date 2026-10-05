"""Generation backends, all of them open-weight models behind one interface.

Three providers are implemented and chosen by ``llm.provider`` in config.yaml.
Every one of them runs a model whose weights are publicly downloadable, so the
system has no dependency on a closed model it cannot inspect or replace.

``groq`` (default via ``auto``)
    Serves ``openai/gpt-oss-120b``, Apache-2.0 licensed open weights, on Groq's
    hardware. A 120B mixture-of-experts model follows the multi-source synthesis
    and citation instructions far more reliably than anything that fits on a
    laptop, and it answers in well under a second, which is what makes the
    branch-then-fuse design usable interactively. It needs a free API key.

``ollama``
    Any locally served open model, default ``qwen2.5:3b-instruct``. Fully
    offline and private, at the cost of a weaker model. Spoken to over plain
    HTTP rather than through the ``ollama`` package, so it adds no dependency.

``local_hf``
    ``Qwen2.5-1.5B-Instruct`` through transformers. The zero-configuration
    fallback: no key, no daemon, nothing to install beyond requirements.txt. It
    is the weakest of the three at multi-document synthesis and is present so
    the project is never unrunnable.

``auto`` resolves in that order, preferring quality when a key is present and
degrading to something that always works. The abstraction is deliberately thin
-- one ``complete`` call -- because that is the whole surface the pipeline needs,
and a thin seam is what lets the evaluation harness swap in a stub.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from functools import lru_cache

from ..config import Config, LLMConfig

logger = logging.getLogger(__name__)

# Some open reasoning models emit their scratchpad inline in the reply. Groq's
# gpt-oss returns it on a separate field, but a locally served Qwen3 or
# DeepSeek-R1 will not, and that text must never reach the user or a citation
# parser.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "").strip()


class LLMUnavailable(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str: ...

    @property
    def description(self) -> str:
        return f"{self.name}:{self.model}"


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, config: LLMConfig) -> None:
        self.model = config.groq_model
        self.fallback_model = config.groq_fallback_model
        self._client = None
        self._api_key = os.environ.get("GROQ_API_KEY", "").strip()

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMUnavailable("GROQ_API_KEY is not set")
            from groq import Groq

            self._client = Groq(api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for model in (self.model, self.fallback_model):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                if content.strip():
                    return strip_reasoning(content)
                # A reasoning model can spend the entire token budget thinking and
                # return empty content. Retrying on the smaller model is cheaper
                # than failing the query.
                logger.warning("model %s returned empty content", model)
            except Exception as error:
                logger.warning("groq model %s failed: %s", model, error)
            if model == self.fallback_model:
                break
        raise LLMUnavailable("all configured groq models failed")


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, config: LLMConfig) -> None:
        self.model = config.ollama_model
        self.host = os.environ.get("OLLAMA_HOST", config.ollama_host).rstrip("/")

    @property
    def available(self) -> bool:
        return self._probe(self.host)

    @staticmethod
    @lru_cache(maxsize=4)
    def _probe(host: str) -> bool:
        """Check once whether a daemon is listening; cached to keep startup fast."""
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=1.5) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            raise LLMUnavailable(f"ollama request failed: {error}") from error
        return strip_reasoning(body.get("message", {}).get("content", ""))


class LocalHFProvider(LLMProvider):
    name = "local_hf"

    def __init__(self, config: LLMConfig) -> None:
        self.model = config.local_hf_model
        self._pipeline = None
        self._failed = False

    @property
    def available(self) -> bool:
        # Reported available without loading weights: the download is large and
        # must not be triggered by a capability probe on app startup.
        return not self._failed

    def _ensure_pipeline(self):
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline

                device = 0 if torch.cuda.is_available() else -1
                logger.info(
                    "loading %s on %s (first run downloads weights)",
                    self.model,
                    "gpu" if device == 0 else "cpu",
                )
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    device=device,
                    dtype=torch.float16 if device == 0 else torch.float32,
                )
            except Exception as error:
                self._failed = True
                raise LLMUnavailable(f"could not load {self.model}: {error}") from error
        return self._pipeline

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        generator = self._ensure_pipeline()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            output = generator(
                messages,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                return_full_text=False,
            )
        except Exception as error:
            raise LLMUnavailable(f"local generation failed: {error}") from error

        generated = output[0]["generated_text"]
        if isinstance(generated, list):  # chat-formatted return
            generated = generated[-1].get("content", "")
        return strip_reasoning(str(generated))


class NullProvider(LLMProvider):
    """Stands in when no backend can run.

    Retrieval still works without a generator, so the app degrades to an
    evidence browser rather than failing outright. The pipeline checks
    ``available`` and reports the reason instead of raising at import time.
    """

    name = "none"
    model = "unavailable"

    @property
    def available(self) -> bool:
        return False

    def complete(self, prompt, system=None, temperature=0.2, max_tokens=700) -> str:
        raise LLMUnavailable(
            "no generation backend available: set GROQ_API_KEY, start ollama, "
            "or allow the local transformers model to download"
        )


_PROVIDERS = {
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "local_hf": LocalHFProvider,
}

_AUTO_ORDER = ("groq", "ollama", "local_hf")


def build_llm(config: Config) -> LLMProvider:
    """Instantiate the configured provider, resolving ``auto`` by availability."""
    requested = config.llm.provider

    if requested in _PROVIDERS:
        provider = _PROVIDERS[requested](config.llm)
        if not provider.available:
            logger.warning(
                "provider %s was requested but reports unavailable", requested
            )
        return provider

    if requested != "auto":
        raise ValueError(
            f"unknown llm provider {requested!r}; "
            f"expected auto or one of {sorted(_PROVIDERS)}"
        )

    for name in _AUTO_ORDER:
        provider = _PROVIDERS[name](config.llm)
        if provider.available:
            logger.info("auto-selected generation backend: %s", provider.description)
            return provider

    logger.warning("no generation backend available; retrieval only")
    return NullProvider()
