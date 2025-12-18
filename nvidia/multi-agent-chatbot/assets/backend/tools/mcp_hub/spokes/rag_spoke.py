"""RAG spoke with deterministic initialization and health checks."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import HubConfig
from ..tool_errors import ToolError, tool_error


@dataclass
class RagDocument:
    id: str
    text: str
    source: Optional[str] = None


@dataclass
class RagState:
    index_path: Path
    documents: List[RagDocument] = field(default_factory=list)


class RagSpoke:
    """Simple file-backed RAG store with single-flight initialization."""

    def __init__(self, config: HubConfig):
        self.config = config
        self._lock = asyncio.Lock()
        self._state: Optional[RagState] = None
        self._init_error: Optional[ToolError] = None
        self._init_attempts = 0

    async def ensure_ready(self) -> Optional[ToolError]:
        """Initialize the store once, caching any failure."""

        if self._state or self._init_error:
            return self._init_error

        async with self._lock:
            if self._state or self._init_error:
                return self._init_error

            self._init_attempts += 1

            try:
                self.config.rag_index_dir.mkdir(parents=True, exist_ok=True)
                state = RagState(index_path=self.config.rag_index_dir)
                self._load_documents(state)
                self._state = state
                return None
            except Exception as exc:  # pragma: no cover - unexpected init failure
                self._init_error = tool_error(
                    tool="rag.query",
                    error_type="runtime_error",
                    message="Failed to initialize RAG store",
                    detail={"error": str(exc)},
                )
                return self._init_error

    def _load_documents(self, state: RagState) -> None:
        index_file = state.index_path / "index.json"
        if not index_file.exists():
            return

        try:
            payload = json.loads(index_file.read_text())
        except Exception:
            payload = []
        for item in payload:
            try:
                state.documents.append(
                    RagDocument(id=item.get("id", ""), text=item.get("text", ""), source=item.get("source"))
                )
            except Exception:
                continue

    def _persist(self) -> None:
        if not self._state:
            return
        index_file = self._state.index_path / "index.json"
        serializable = [doc.__dict__ for doc in self._state.documents]
        index_file.write_text(json.dumps(serializable, indent=2))

    async def query(self, query: str) -> Dict[str, object]:
        error = await self.ensure_ready()
        if error:
            return error

        assert self._state  # for type checkers
        query_lower = query.lower().strip()
        matches = [doc for doc in self._state.documents if query_lower in doc.text.lower()]

        if not matches:
            return {
                "ok": True,
                "tool": "rag.query",
                "answer": "No indexed documents matched the query.",
                "sources": [],
            }

        top = matches[0]
        return {
            "ok": True,
            "tool": "rag.query",
            "answer": top.text,
            "sources": [top.source] if top.source else [],
        }

    async def ingest(self, text: str, source: Optional[str] = None) -> Dict[str, object]:
        error = await self.ensure_ready()
        if error:
            return error

        assert self._state
        doc = RagDocument(id=str(int(time.time() * 1000)), text=text.strip(), source=source)
        self._state.documents.append(doc)
        self._persist()
        return {
            "ok": True,
            "tool": "rag.ingest",
            "stored": True,
            "count": len(self._state.documents),
        }

    async def health(self) -> Dict[str, object]:
        error = await self.ensure_ready()
        if error:
            return error

        assert self._state
        return {
            "ok": True,
            "tool": "rag.health",
            "documents_indexed": len(self._state.documents),
            "index_path": str(self._state.index_path),
            "init_attempts": self._init_attempts,
        }

    @property
    def init_attempts(self) -> int:
        return self._init_attempts


def register_tools(hub, config: HubConfig):
    spoke = RagSpoke(config)

    def _retag(result: object, tool_name: str):
        if isinstance(result, dict):
            return {**result, "tool": tool_name}
        return result

    async def query_tool(question: str):
        return await spoke.query(question)

    async def ingest_tool(text: str, source: Optional[str] = None):
        return await spoke.ingest(text=text, source=source)

    async def search_documents_tool(question: str):
        return _retag(await query_tool(question), "search_documents")

    async def health_tool():
        return await spoke.health()

    hub.register_tool("rag.query", "Query the local RAG index", query_tool)
    hub.register_tool("rag.ingest", "Ingest content into the RAG index", ingest_tool)
    hub.register_tool("rag.health", "Health check for RAG index", health_tool)
    # Legacy alias so prompts and warmup continue to match hub names
    hub.register_tool("search_documents", "Query the local RAG index", search_documents_tool)

    # Expose spoke for tests
    hub.rag_spoke = spoke  # type: ignore[attr-defined]

