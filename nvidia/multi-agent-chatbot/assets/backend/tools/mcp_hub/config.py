"""Central configuration for the MCP hub and spokes.

All environment variables are parsed in a single place to avoid mismatched
defaults across tools. Secrets are redacted when logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def _resolve_path(value: Optional[str], default: str) -> Path:
    """Resolve storage paths consistently.

    Paths are normalized to absolute paths to avoid dependency on the current
    working directory when the hub is launched from different entrypoints.
    """

    target = value or default
    return Path(target).expanduser().resolve()


def _derive_endpoint(explicit: Optional[str], base_url: Optional[str], path: str) -> Optional[str]:
    """Prefer explicitly configured endpoints but fall back to service URLs."""

    if explicit:
        return explicit
    if base_url:
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    return None


@dataclass
class HubConfig:
    """Configuration for all MCP spokes."""

    tavily_api_key: Optional[str]
    tavily_endpoint: Optional[str]
    tavily_timeout: float

    rag_index_dir: Path

    flux_endpoint: Optional[str]
    flux_timeout: float

    wan_endpoint: Optional[str]
    wan_timeout: float

    ministral_endpoint: Optional[str]
    ministral_timeout: float

    @classmethod
    def from_env(cls) -> "HubConfig":
        """Create a configuration object from environment variables."""

        # Prefer explicit service URLs but fall back to the docker-compose defaults so
        # the tools are wired up even when the environment is minimally configured.
        flux_service_url = os.getenv("FLUX_SERVICE_URL", "http://flux-service:8080")
        video_service_url = os.getenv("VIDEO_SERVICE_URL", "http://video-service:8081")

        ministral_service_url = os.getenv("MINISTRAL_SERVICE_URL", "http://backend:8000")

        return cls(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            tavily_endpoint=os.getenv("TAVILY_ENDPOINT"),
            tavily_timeout=float(os.getenv("TAVILY_TIMEOUT", "15")),
            rag_index_dir=_resolve_path(os.getenv("RAG_INDEX_DIR"), "~/.cache/mcp_rag_index"),
            flux_endpoint=_derive_endpoint(os.getenv("FLUX_ENDPOINT"), flux_service_url, "generate_image"),
            flux_timeout=float(os.getenv("FLUX_TIMEOUT", "60")),
            wan_endpoint=_derive_endpoint(os.getenv("WAN_ENDPOINT"), video_service_url, "generate_video"),
            wan_timeout=float(os.getenv("WAN_TIMEOUT", "60")),
            ministral_endpoint=_derive_endpoint(
                os.getenv("MINISTRAL_ENDPOINT"), ministral_service_url, "vision/ministral/describe"
            ),
            ministral_timeout=float(os.getenv("MINISTRAL_TIMEOUT", "30")),
        )

    def summary(self) -> Dict[str, object]:
        """Return a redacted, structured configuration summary for logging."""

        return {
            "tavily_endpoint": self.tavily_endpoint or "disabled",
            "tavily_api_key_present": bool(self.tavily_api_key),
            "rag_index_dir": str(self.rag_index_dir),
            "flux_endpoint": self.flux_endpoint or "disabled",
            "wan_endpoint": self.wan_endpoint or "disabled",
            "ministral_endpoint": self.ministral_endpoint or "disabled",
            "timeouts": {
                "tavily": self.tavily_timeout,
                "flux": self.flux_timeout,
                "wan": self.wan_timeout,
                "ministral": self.ministral_timeout,
            },
        }

