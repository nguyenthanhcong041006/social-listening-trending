"""
LLM Client Utility for Offline Sports Listening Tasks
Supports Google Gemini REST API and OpenAI-compatible endpoints (Groq, OpenRouter, DeepSeek, Ollama).
Requires only 'requests' (zero heavy SDK dependencies).
"""

import os
import json
import re
import time
from typing import Optional, Dict, Any, List, Union
import requests
from loguru import logger


def load_env_file(env_path: str = ".env") -> None:
    """Lightweight .env loader that populates os.environ without requiring python-dotenv."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        logger.debug(f"Failed to load .env file from {env_path}: {e}")


def clean_json_string(raw: str) -> str:
    """Strips markdown code blocks, trailing commas, and returns clean JSON string."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    # Remove leading/trailing non-json characters if any
    first_bracket = min(
        [pos for pos in [text.find("{"), text.find("[")] if pos != -1] or [0]
    )
    last_bracket = max(
        [pos for pos in [text.rfind("}"), text.rfind("]")] if pos != -1] or [len(text) - 1]
    )
    return text[first_bracket : last_bracket + 1]


class LLMClient:
    """
    Robust REST-based LLM client supporting Gemini and OpenAI-compatible providers.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        load_env_file()
        
        # Auto-detect provider if not explicitly given
        gemini_key = api_key or os.getenv("GEMINI_API_KEY")
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        
        env_provider = os.getenv("LLM_PROVIDER", "").lower()
        if provider:
            self.provider = provider.lower()
        elif env_provider in ["gemini", "openai"]:
            self.provider = env_provider
        elif gemini_key:
            self.provider = "gemini"
        elif openai_key:
            self.provider = "openai"
        else:
            self.provider = "none"

        self.timeout = timeout

        if self.provider == "gemini":
            self.api_key = gemini_key
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        elif self.provider == "openai":
            self.api_key = openai_key
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        else:
            self.api_key = None
            self.model = None
            self.base_url = None

    def is_available(self) -> bool:
        """Returns True if a valid provider and API key are configured."""
        return bool(self.api_key and self.provider in ["gemini", "openai"])

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        """Generates plain text response."""
        if not self.is_available():
            raise ValueError(
                "LLMClient is not configured. Please set GEMINI_API_KEY or OPENAI_API_KEY in your environment or .env file."
            )

        if self.provider == "gemini":
            return self._call_gemini(prompt, system_prompt, temperature=temperature, response_mime="text/plain")
        else:
            return self._call_openai(prompt, system_prompt, temperature=temperature, json_mode=False)

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> Union[Dict, List]:
        """Generates structured JSON response and parses it."""
        if not self.is_available():
            raise ValueError(
                "LLMClient is not configured. Please set GEMINI_API_KEY or OPENAI_API_KEY in your environment or .env file."
            )

        if self.provider == "gemini":
            raw = self._call_gemini(prompt, system_prompt, temperature=temperature, response_mime="application/json")
        else:
            raw = self._call_openai(prompt, system_prompt, temperature=temperature, json_mode=True)

        cleaned = clean_json_string(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}\nRaw output: {raw[:300]}")
            raise

    def _call_gemini(
        self, prompt: str, system_prompt: Optional[str], temperature: float, response_mime: str, max_retries: int = 3
    ) -> str:
        endpoint = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will strictly follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": response_mime,
            },
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(endpoint, json=body, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    raise ValueError(f"Empty or blocked response from Gemini API: {data}")
                elif resp.status_code == 429:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Gemini API rate limit (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Gemini API error HTTP {resp.status_code}: {resp.text[:300]}")
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)

        raise RuntimeError("Gemini API calls exceeded max retries.")

    def _call_openai(
        self, prompt: str, system_prompt: Optional[str], temperature: float, json_mode: bool, max_retries: int = 3
    ) -> str:
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                resp = requests.post(endpoint, headers=headers, json=body, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0]["message"].get("content", "")
                    raise ValueError(f"Empty choices in OpenAI response: {data}")
                elif resp.status_code == 429:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"OpenAI API rate limit (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"OpenAI API error HTTP {resp.status_code}: {resp.text[:300]}")
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)

        raise RuntimeError("OpenAI API calls exceeded max retries.")

