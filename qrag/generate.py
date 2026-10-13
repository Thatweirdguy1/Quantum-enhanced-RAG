"""Grounded answer generation through the configured local Ollama daemon."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import GenConfig
from .security import build_context, validate_query


@dataclass
class GeneratedAnswer:
    text: str
    context_report: dict


class OllamaGenerator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.url = cfg.ollama_url.rstrip("/") + "/api/generate"

    def generate(self, query: str, passages: list[str],
                 doc_ids: list[str]) -> GeneratedAnswer:
        query = validate_query(query)
        context, report = build_context(
            passages[: self.cfg.max_context_docs],
            doc_ids=doc_ids[: self.cfg.max_context_docs],
        )
        prompt = (
            "Answer the question using only the numbered context. "
            "Cite supporting passages as [1], [2], etc. If the context is "
            "insufficient, say so.\n\n"
            f"Question: {query}\n\n{context}"
        )
        response = requests.post(
            self.url,
            json={"model": self.cfg.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": self.cfg.temperature}},
            timeout=300,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        if not text:
            raise RuntimeError("generator returned an empty response")
        return GeneratedAnswer(text, report)
