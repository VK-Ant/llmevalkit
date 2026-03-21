"""LLM client abstraction supporting multiple providers."""

from __future__ import annotations

import json
import os
import sys

from tenacity import retry, stop_after_attempt, wait_exponential

from llmevalkit.models import EvalConfig, Provider


class LLMClient:
    """Unified LLM client for OpenAI, Azure, Anthropic, Groq, Ollama,
    HuggingFace, and any OpenAI-compatible API."""

    def __init__(self, config):
        self.config = config
        self._client = None
        self._provider_type = None
        self._supports_json_mode = False
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
        elif provider in (Provider.HUGGINGFACE, "huggingface"):
            self._setup_huggingface()
        elif provider in (Provider.CUSTOM, "custom"):
            self._setup_custom()
        else:
            raise ValueError("Unsupported provider: {}".format(provider))

    def _setup_openai(self):
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key.")
        self._client = OpenAI(api_key=api_key, timeout=self.config.timeout)
        self._provider_type = "openai"
        self._supports_json_mode = True

    def _setup_azure(self):
        from openai import AzureOpenAI
        api_key = self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        base_url = self.config.base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = self.config.api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-01"
        )
        if not api_key:
            raise ValueError(
                "Azure needs api_key. Set AZURE_OPENAI_API_KEY or pass api_key."
            )
        if not base_url:
            raise ValueError(
                "Azure needs base_url. Set AZURE_OPENAI_ENDPOINT or pass base_url. "
                "Example: https://your-resource.openai.azure.com/"
            )
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            timeout=self.config.timeout,
        )
        self._provider_type = "openai"
        self._supports_json_mode = True

    def _setup_anthropic(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic not installed. Run: pip install llmevalkit[anthropic]"
            )
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key.")
        # Anthropic uses httpx timeout, pass as float (seconds).
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=float(self.config.timeout),
        )
        self._provider_type = "anthropic"
        self._supports_json_mode = False

    def _setup_groq(self):
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY or pass api_key.")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=self.config.timeout,
        )
        self._provider_type = "openai"
        self._supports_json_mode = False

    def _setup_ollama(self):
        from openai import OpenAI
        base_url = self.config.base_url or "http://localhost:11434/v1"
        self._client = OpenAI(
            api_key="ollama",
            base_url=base_url,
            timeout=self.config.timeout,
        )
        self._provider_type = "openai"
        self._supports_json_mode = False

    def _setup_huggingface(self):
        """HuggingFace Inference API (serverless or dedicated endpoints).

        Uses the OpenAI-compatible API that HuggingFace provides.
        Set HF_API_KEY or pass api_key.
        Optionally set base_url for dedicated endpoints.
        """
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "Set HF_API_KEY (or HF_TOKEN) or pass api_key. "
                "Get your token from https://huggingface.co/settings/tokens"
            )
        base_url = self.config.base_url or "https://api-inference.huggingface.co/v1"
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.timeout,
        )
        self._provider_type = "openai"
        self._supports_json_mode = False

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
        self._supports_json_mode = False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(self, prompt, system="", json_mode=False):
        """Generate a completion from the LLM."""
        if self._provider_type == "openai":
            return self._generate_openai(prompt, system, json_mode)
        elif self._provider_type == "anthropic":
            return self._generate_anthropic(prompt, system)
        else:
            raise ValueError("Unknown provider type: {}".format(self._provider_type))

    def _generate_openai(self, prompt, system, json_mode):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        if json_mode and self._supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty response. Check your model name and API key.")
        return content

    def _generate_anthropic(self, prompt, system):
        kwargs = {
            "model": self.config.model,
            "max_tokens": 4096,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        if not response.content:
            raise ValueError("Anthropic returned empty response.")
        return response.content[0].text

    def generate_json(self, prompt, system=""):
        """Generate a response and parse it as JSON."""
        if self._supports_json_mode:
            raw = self._generate_openai(prompt, system, json_mode=True)
        else:
            json_instruction = (
                "\n\nIMPORTANT: Respond with ONLY a valid JSON object. "
                "No markdown, no backticks, no extra text before or after the JSON."
            )
            raw = self.generate(prompt + json_instruction, system=system)

        raw = raw.strip()

        # Remove markdown code fences.
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        # Find JSON object in response.
        if not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end + 1]

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                "Could not parse LLM response as JSON. "
                "Raw response: {}... Error: {}".format(raw[:200], e)
            )
