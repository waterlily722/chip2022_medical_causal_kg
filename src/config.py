#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 32768
CONFIG_FILE_NAME = ".env"


@dataclass(frozen=True)
class QwenSettings:
    api_key: str | None
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    config_path: str
    configured: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "config_path": self.config_path,
            "configured": self.configured,
        }


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_qwen_settings() -> QwenSettings:
    """Return Qwen settings from .env (or environment variables)."""
    root = Path(__file__).resolve().parents[1]
    env_path = root / CONFIG_FILE_NAME
    if load_dotenv:
        load_dotenv(env_path)

    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL") or DEFAULT_BASE_URL
    model = os.getenv("QWEN_MODEL") or DEFAULT_MODEL
    temperature = _coerce_float(os.getenv("QWEN_TEMPERATURE"), DEFAULT_TEMPERATURE)
    max_tokens = _coerce_int(os.getenv("QWEN_MAX_TOKENS"), DEFAULT_MAX_TOKENS)

    config_path = str(env_path)
    return QwenSettings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        config_path=config_path,
        configured=bool(api_key),
    )


def get_qwen_settings() -> Dict[str, Any]:
    """Backward-compatible dict accessor."""
    return load_qwen_settings().as_dict()
