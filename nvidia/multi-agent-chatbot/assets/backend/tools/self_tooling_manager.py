#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility helpers for creating, validating, and executing self-authored tools.

This module lets the LLM author lightweight tools that are persisted to disk and
executed under tight safety controls. Tools are saved to JSON files inside the
``self_tooling`` directory and executed without invoking a shell, which limits
command injection opportunities.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class SelfTool:
    """A persisted tool definition owned by the LLM."""

    name: str
    description: str
    commands: List[List[str]]
    created_at: float = field(default_factory=lambda: time.time())


class SelfToolingManager:
    """Manage creation, validation, and execution of self-authored tools."""

    MAX_COMMANDS = 10
    MAX_ARGS_PER_COMMAND = 20

    # Conservative allow list of entrypoint binaries. Extend cautiously.
    ALLOWED_BINARIES = {
        "ssh",
        "docker",
        "apt",
        "apt-get",
        "systemctl",
        "nvidia-smi",
        "mkdir",
        "ls",
        "echo",
        "touch",
        "cat",
        "scp",
    }

    FORBIDDEN_SUBSTRINGS = {
        "rm -rf",
        "mkfs",
        "dd if=",
        "shutdown",
        "poweroff",
        "halt",
        ":(){",
    }

    # Remote commands executed via SSH use the same binary allow list, plus sudo
    # when the remote host requires elevated privileges for package installation.
    REMOTE_ALLOWED_BINARIES = ALLOWED_BINARIES | {"sudo", "reboot"}

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _tool_path(self, name: str) -> Path:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", name).strip("-")
        if not safe_name:
            raise ValueError("Tool name must contain at least one alphanumeric character.")
        return self.base_dir / f"{safe_name}.json"

    def _validate_commands(self, commands: Iterable[Iterable[str]]) -> List[List[str]]:
        commands_list = [list(cmd) for cmd in commands]
        if not commands_list:
            raise ValueError("At least one command is required.")
        if len(commands_list) > self.MAX_COMMANDS:
            raise ValueError(f"No more than {self.MAX_COMMANDS} commands per tool are allowed.")

        for cmd in commands_list:
            if not cmd:
                raise ValueError("Commands cannot be empty.")
            if len(cmd) > self.MAX_ARGS_PER_COMMAND:
                raise ValueError(
                    f"Commands cannot exceed {self.MAX_ARGS_PER_COMMAND} arguments (got {len(cmd)})."
                )

            joined = " ".join(cmd).lower()
            for bad in self.FORBIDDEN_SUBSTRINGS:
                if bad in joined:
                    raise ValueError(f"Command rejected because it contains a dangerous pattern: {bad}")

            binary = cmd[0]
            if binary not in self.ALLOWED_BINARIES:
                raise ValueError(
                    f"Command entrypoint '{binary}' is not allowed. Allowed: {sorted(self.ALLOWED_BINARIES)}"
                )

            if binary == "ssh":
                self._validate_ssh_command(cmd)

        return commands_list

    def _validate_ssh_command(self, command: List[str]) -> None:
        if len(command) < 3:
            raise ValueError("SSH commands must include a target host and a remote command to run.")

        remote_command = " ".join(command[2:])
        try:
            remote_tokens = shlex.split(remote_command)
        except ValueError as exc:
            raise ValueError(f"Unable to parse remote command: {exc}") from exc

        if not remote_tokens:
            raise ValueError("Remote command cannot be empty.")

        remote_binary = remote_tokens[0]
        if remote_binary not in self.REMOTE_ALLOWED_BINARIES:
            raise ValueError(
                f"Remote command entrypoint '{remote_binary}' is not allowed over SSH. Allowed:"
                f" {sorted(self.REMOTE_ALLOWED_BINARIES)}"
            )

        joined_remote = " ".join(remote_tokens).lower()
        for bad in self.FORBIDDEN_SUBSTRINGS:
            if bad in joined_remote:
                raise ValueError(
                    f"Remote command rejected because it contains a dangerous pattern: {bad}"
                )

    def save_tool(self, name: str, description: str, commands: Iterable[Iterable[str]]) -> SelfTool:
        validated_commands = self._validate_commands(commands)
        tool = SelfTool(name=name, description=description, commands=validated_commands)
        tool_path = self._tool_path(name)
        tool_path.write_text(json.dumps(asdict(tool), indent=2))
        return tool

    def load_tool(self, name: str) -> SelfTool:
        tool_path = self._tool_path(name)
        if not tool_path.exists():
            raise FileNotFoundError(f"No tool named '{name}' exists in self_tooling.")
        raw = json.loads(tool_path.read_text())
        return SelfTool(**raw)

    def list_tools(self) -> List[Dict[str, str]]:
        tools = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                tools.append({
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "created_at": data.get("created_at", 0),
                })
            except Exception:
                continue
        return tools

    def run_tool(self, name: str, timeout: int = 120) -> Dict[str, object]:
        tool = self.load_tool(name)
        results = []

        for cmd in tool.commands:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            results.append(
                {
                    "command": cmd,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )
            if proc.returncode != 0:
                break

        return {
            "tool": tool.name,
            "description": tool.description,
            "results": results,
        }


__all__ = ["SelfTool", "SelfToolingManager"]
