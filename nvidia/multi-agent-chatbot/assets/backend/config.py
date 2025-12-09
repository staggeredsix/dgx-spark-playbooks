#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""ConfigManager for managing the configuration of the chat application."""

import json
import os
import logging
import threading
from typing import List, Optional

from logger import logger
from models import ChatConfig


class ConfigManager:
    def __init__(self, config_path: str):
        """Initialize the ConfigManager"""
        self.config_path = config_path
        self.config = None
        self._last_modified = 0
        self._lock = threading.Lock()
        self._ensure_config_exists()
        self.read_config()

    @property
    def default_flux_model(self) -> str:
        return "black-forest-labs/FLUX.1-dev-onnx/transformer.opt/fp4"
    
    def _ensure_config_exists(self) -> None:
        """Ensure config.json exists, creating it with default values if not."""
        models = []
        models = os.getenv("MODELS", "")

        if models:
            models = [model.strip() for model in models.split(",") if model.strip()]
        else:
            logger.warning("MODELS environment variable not set, using empty models list")

        if not os.path.exists(self.config_path):
            logger.debug(f"Config file {self.config_path} not found, creating default config")
            default_config = ChatConfig(
                sources=[],
                models=models,
                selected_model=models[0] if models else None,
                selected_sources=[],
                current_chat_id=None,
                tavily_enabled=False,
                tavily_api_key=None,
                supervisor_model=models[0] if models else None,
                code_model=os.getenv("CODE_MODEL", "qwen3-coder:30b"),
                vision_model=os.getenv("VISION_MODEL", "ministral-3:14b"),
                flux_enabled=False,
                flux_model=self.default_flux_model,
                hf_api_key=None,
            )
            
            with open(self.config_path, "w") as f:
                json.dump(default_config.model_dump(), f, indent=2)
        else:
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                existing_config = ChatConfig(**data)

                if models:
                    existing_config.models = models
                    if not existing_config.selected_model or existing_config.selected_model not in models:
                        existing_config.selected_model = models[0]

                if not existing_config.supervisor_model:
                    existing_config.supervisor_model = existing_config.selected_model
                if not existing_config.code_model:
                    existing_config.code_model = os.getenv("CODE_MODEL", "qwen3-coder:30b")
                if not existing_config.vision_model:
                    existing_config.vision_model = os.getenv("VISION_MODEL", "ministral-3:14b")
                if existing_config.flux_model is None:
                    existing_config.flux_model = self.default_flux_model
                
                with open(self.config_path, "w") as f:
                    json.dump(existing_config.model_dump(), f, indent=2)
                    
                logger.debug(f"Updated existing config with models: {models}")
            except Exception as e:
                logger.error(f"Error updating existing config: {e}")
                default_config = ChatConfig(
                    sources=[],
                    models=models,
                    selected_model=models[0] if models else None,
                    selected_sources=[],
                    current_chat_id=None,
                    tavily_enabled=False,
                    tavily_api_key=None,
                    supervisor_model=models[0] if models else None,
                    code_model=os.getenv("CODE_MODEL", "qwen3-coder:30b"),
                    vision_model=os.getenv("VISION_MODEL", "ministral-3:14b"),
                    flux_enabled=False,
                    flux_model=self.default_flux_model,
                    hf_api_key=None,
                )
                with open(self.config_path, "w") as f:
                    json.dump(default_config.model_dump(), f, indent=2)
    
    def read_config(self) -> ChatConfig:
        """Read config from file, but only if it has changed since last read."""
        with self._lock:
            try:
                current_mtime = os.path.getmtime(self.config_path)
                if self.config is None or current_mtime > self._last_modified:
                    with open(self.config_path, "r") as f:
                        data = json.load(f)
                    self.config = ChatConfig(**data)
                    self._last_modified = current_mtime
                return self.config
            except Exception as e:
                logger.error(f"Error reading config: {e}")
                if self.config is None:
                    models = []
                    models = os.getenv("MODELS", "")
                    if models:
                        models = [model.strip() for model in models.split(",") if model.strip()]
                    
                    self.config = ChatConfig(
                        sources=[],
                        models=models,
                        selected_model=models[0] if models else "gpt-oss-120b",
                        selected_sources=[],
                        current_chat_id="1",
                        tavily_enabled=False,
                        tavily_api_key=None,
                        supervisor_model=models[0] if models else None,
                        code_model=os.getenv("CODE_MODEL", "qwen3-coder:30b"),
                        vision_model=os.getenv("VISION_MODEL", "ministral-3:14b"),
                        flux_enabled=False,
                        flux_model=self.default_flux_model,
                        hf_api_key=None,
                    )
                return self.config

    def write_config(self, new_config: ChatConfig) -> None:
        """Thread-safe write config to file."""
        with self._lock:
            with open(self.config_path, "w") as f:
                json.dump(new_config.model_dump(), f, indent=2)
            self.config = new_config
            self._last_modified = os.path.getmtime(self.config_path)

    def get_sources(self) -> List[str]:
        """Return list of available sources."""
        self.config = self.read_config()
        return self.config.sources
    
    def get_selected_sources(self) -> List[str]:
        """Return list of selected sources."""
        self.config = self.read_config()
        return self.config.selected_sources
    
    def get_available_models(self) -> List[str]:    
        """Return list of available models."""
        self.config = self.read_config()
        return self.config.models
    
    def get_selected_model(self) -> str:
        """Return the selected model."""
        self.config = self.read_config()
        logger.debug(f"Selected model: {self.config.selected_model}")
        return self.config.selected_model

    def get_supervisor_model(self) -> Optional[str]:
        self.config = self.read_config()
        return self.config.supervisor_model or self.config.selected_model

    def get_code_model(self) -> Optional[str]:
        self.config = self.read_config()
        return self.config.code_model

    def get_vision_model(self) -> Optional[str]:
        self.config = self.read_config()
        return self.config.vision_model
    
    def get_current_chat_id(self) -> str:
        """Return the current chat id."""
        self.config = self.read_config()
        return self.config.current_chat_id

    def get_tavily_settings(self) -> dict:
        """Return the Tavily enablement flag and API key."""
        self.config = self.read_config()
        return {
            "enabled": bool(self.config.tavily_enabled),
            "api_key": self.config.tavily_api_key,
        }
    
    
    def updated_selected_sources(self, new_sources: List[str]) -> None:
        """Update the selected sources in the config."""
        self.config = self.read_config().model_copy(update={"selected_sources": new_sources})
        self.write_config(self.config)
    
    def updated_selected_model(self, new_model: str) -> None:
        """Update the selected model in the config."""
        self.config = self.read_config().model_copy(update={"selected_model": new_model})
        logger.debug(f"Updated selected model to: {new_model}")
        self.write_config(self.config)
    
    def updated_current_chat_id(self, new_chat_id: str) -> None:
        """Update the current chat id in the config."""
        self.config = self.read_config().model_copy(update={"current_chat_id": new_chat_id})
        self.write_config(self.config)

    def update_tavily_settings(self, enabled: bool, api_key: str | None) -> None:
        """Update Tavily enablement and API key settings in the config."""
        self.config = self.read_config().model_copy(
            update={
                "tavily_enabled": enabled,
                "tavily_api_key": api_key,
            }
        )
        self.write_config(self.config)

    def get_model_settings(self) -> dict:
        config = self.read_config()
        return {
            "supervisor_model": config.supervisor_model or config.selected_model,
            "code_model": config.code_model,
            "vision_model": config.vision_model,
            "flux_enabled": bool(config.flux_enabled),
            "flux_model": config.flux_model or self.default_flux_model,
            "hf_api_key": config.hf_api_key,
        }

    def update_model_settings(
        self,
        supervisor_model: Optional[str] = None,
        code_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        flux_enabled: Optional[bool] = None,
        flux_model: Optional[str] = None,
        hf_api_key: Optional[str] = None,
    ) -> None:
        config = self.read_config()

        updates = {}
        if supervisor_model:
            updates["supervisor_model"] = supervisor_model
            updates["selected_model"] = supervisor_model
        if code_model:
            updates["code_model"] = code_model
        if vision_model:
            updates["vision_model"] = vision_model
        if flux_enabled is not None:
            updates["flux_enabled"] = flux_enabled
        if flux_model:
            updates["flux_model"] = flux_model
        if hf_api_key is not None:
            updates["hf_api_key"] = hf_api_key

        new_config = config.model_copy(update=updates)

        # Keep track of models seen for convenience in dropdowns
        known_models = set(new_config.models or [])
        if supervisor_model:
            known_models.add(supervisor_model)
        new_config.models = list(known_models)

        self.write_config(new_config)
