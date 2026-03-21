"""LLM client abstraction supporting multiple providers."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from llmevalkit.models import EvalConfig, Provider


class LLMClient:
    """Unified LLM client that abstracts away provider differences.
    
    Supports: OpenAI, Azure OpenAI, Anthropic, Groq, Ollama, and any OpenAI-compatible API.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self._client = None
        self._setup_client()

    def _setup_client(self):
        provider = self.config.provider
        
        if provider in (Provider.OPENAI, "openai"):
            self._setup_openai()
        elif provider in (Provider.AZURE, "azure"):
            self._setup_azure()
        elif provider in (Provider.ANTHROPIC, "anthropic"):
            self._setup_anthropic()
        elif provider in (Provider.GROQ, "groq"):
            self._setup_groq()
        elif provider in (Provider.OLLAMA, "ollama"):
            self._setup_ollama()
        elif provider in (Provider.CUSTOM, "custom"):
            self._setup_custom()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _setup_openai(self):
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
        self._client = OpenAI(api_key=api_key, timeout=self.config.timeout)
        self._provider_type = "openai"

    def _setup_azure(self):
        from openai import AzureOpenAI
        api_key = self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        base_url = self.config.base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = self.config.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if not api_key or not base_url:
            raise ValueError("Azure requires api_key and base_url (or env vars AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)")
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            timeout=self.config.timeout,
        )
        self._provider_type = "openai"  # Azure uses OpenAI SDK

    def _setup_anthropic(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install llmeval[anthropic]")
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key.")
        self._client = anthropic.Anthropic(api_key=api_key, timeout=self.config.timeout)
        self._provider_type = "anthropic"

    def _setup_groq(self):
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY or pass api_key.")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=self.config.timeout,
        )
        self._provider_type = "openai"

    def _setup_ollama(self):
        from openai import OpenAI
        base_url = self.config.base_url or "http://localhost:11434/v1"
        self._client = OpenAI(
            api_key="ollama",
            base_url=base_url,
            timeout=self.config.timeout,
        )
        self._provider_type = "openai"

    def _setup_custom(self):
        from openai import OpenAI
        if not self.config.base_url:
            raise ValueError("Custom provider requires base_url.")
        self._client = OpenAI(
            api_key=self.config.api_key or "custom",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self._provider_type = "openai"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> str:
        """Generate a completion from the LLM."""
        if self._provider_type == "openai":
            return self._generate_openai(prompt, system, json_mode)
        elif self._provider_type == "anthropic":
            return self._generate_anthropic(prompt, system)
        else:
            raise ValueError(f"Unknown provider type: {self._provider_type}")

    def _generate_openai(self, prompt: str, system: str, json_mode: bool) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _generate_anthropic(self, prompt: str, system: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": 4096,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def generate_json(self, prompt: str, system: str = "") -> dict:
        """Generate and parse JSON response."""
        if self._provider_type == "openai":
            raw = self._generate_openai(prompt, system, json_mode=True)
        else:
            raw = self.generate(prompt + "\n\nRespond ONLY with valid JSON, no markdown.", system)
        
        # Clean up potential markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        
        return json.loads(raw)
