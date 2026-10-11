"""LLM providers + committed response cache. Owner: P4.

Defaults come from config/settings.yaml `llm:`; env vars (also read from the git-ignored .env) override:
  LLM_PROVIDER      auto | gemini | groq | anthropic | offline
                    auto = gemini if GEMINI_API_KEY is set, else groq if GROQ_API_KEY is set,
                    else anthropic if ANTHROPIC_API_KEY is set, else offline (cache + template only).
                    With both Gemini and Groq keys, Groq serves requests Gemini cannot (quota, outages).
  GEMINI_API_KEY    Google AI Studio key
  GEMINI_MODELS     comma-separated fallback chain (default: settings llm.model + llm.fallback_models)
  GROQ_API_KEY      Groq key (OpenAI-compatible API)
  GROQ_MODELS       comma-separated chain (default: settings llm.groq_models)
  ANTHROPIC_API_KEY / LLM_MODEL (default claude-opus-5-5) / LLM_EFFORT (default low)
  LLM_MIN_INTERVAL_S  seconds between API calls (free-tier rate limits)
  LLM_CACHE_DIR     default settings llm.cache_dir (data/llm_cache, committed: D10)

Reproducibility comes from the cache: the same (provider, primary model, system, message, schema)
always returns the stored response, so judges can rerun everything without an API key (D10).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from threading import RLock
from pathlib import Path

from src.common import config
from src.common.config import ROOT

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def llm_settings() -> dict:
    return config.settings().get("llm") or {}

log = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


def default_gemini_models() -> list[str]:
    env = os.getenv("GEMINI_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    s = llm_settings()
    return [m for m in [s.get("model"), *(s.get("fallback_models") or [])] if m] or ["gemini-3.7-flash"]


class LLMUnavailable(RuntimeError):
    """The provider could not produce a usable response; the caller should fall back to the template."""


class ResponseCache:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory or os.getenv("LLM_CACHE_DIR") or ROOT / llm_settings().get("cache_dir", "data/llm_cache"))

    @staticmethod
    def key(*parts) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        path = self.dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, value: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def parse_json_text(text: str | None) -> dict:
    if not text:
        raise LLMUnavailable("empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())  # some open models wrap JSON in fences
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMUnavailable("response was not valid JSON") from e


class CachedProvider:
    """Shared cache lookup / store. Subclasses implement _call() -> (output, meta)."""
    name = "base"

    def __init__(self, cache: ResponseCache | None = None):
        self.cache = cache or ResponseCache()
        self.last_meta: dict = {}

    def cache_identity(self) -> list:
        return [self.name, self.model]

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        key = self.cache.key(*self.cache_identity(), system, user, schema)
        cached = self.cache.get(key)
        if cached is not None:
            self.last_meta = {**cached.get("_meta", {}), "cache_hit": True}
            return cached["output"]
        start = time.perf_counter()
        output, meta = self._call(system, user, schema)
        meta = {**meta, "latency_s": round(time.perf_counter() - start, 2)}
        self.cache.put(key, {"output": output, "_meta": meta})
        self.last_meta = {**meta, "cache_hit": False}
        return output

    def _call(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        raise NotImplementedError


class OfflineProvider:
    name = "offline"
    model = "none"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raise LLMUnavailable("offline mode")


class CacheOnlyProvider(CachedProvider):
    """Replay the configured provider's committed responses without an SDK or key."""
    def __init__(self, provider: str, model: str, effort: str | None = None):
        super().__init__()
        self.name, self.model, self.effort = provider + '-cache-only', model, effort
        self.provider_name = provider

    def cache_identity(self):
        return [self.provider_name, self.model] + ([self.effort] if self.effort is not None else [])

    def _call(self, system, user, schema):
        raise LLMUnavailable('No matching committed response; live provider unavailable')


class GeminiProvider(CachedProvider):
    """Google Gemini / Gemma via the google-genai SDK. Tries each model in the chain in order, because
    free-tier models are intermittently overloaded (503) or rate limited (429)."""
    name = "gemini"
    RETRYABLE = {429, 500, 503, 504}
    BACKOFF_S = (3,)          # one retry per model, then move down the chain
    TIMEOUT_MS = 45_000       # overloaded models can hang; fail fast and try the next one
    QUOTA_REST_S = 600        # skip a quota-exhausted model for 10 minutes

    def __init__(self, models: list[str] | None = None, cache: ResponseCache | None = None):
        super().__init__(cache)
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as e:
            raise LLMUnavailable("google-genai package not installed") from e
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMUnavailable("GEMINI_API_KEY not set")
        self._errors, self._types = errors, types
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=self.TIMEOUT_MS))
        self.models = models or default_gemini_models()
        self.model = self.models[0]
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL_S") or llm_settings().get("min_interval_s", 1.0))
        self.temperature = llm_settings().get("temperature")
        self._last_call = 0.0
        self._throttle_lock = RLock()
        self._exhausted: dict[str, float] = {}   # model -> monotonic time its quota can be retried

    def cache_identity(self) -> list:
        # Keyed on the primary model only: fallbacks are availability backups, and the model that
        # actually served each response is stored in its metadata.
        return [self.name, self.models[0]]

    def _throttle(self):
        with self._throttle_lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _call(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        extra = {} if self.temperature is None else {"temperature": self.temperature}
        gen_config = self._types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json", response_json_schema=schema, **extra)
        failures = []
        for model in self.models:
            if self._exhausted.get(model, 0) > time.monotonic():
                failures.append(f"{model}: quota exhausted (skipped)")
                continue
            for attempt in range(len(self.BACKOFF_S) + 1):
                self._throttle()
                try:
                    r = self.client.models.generate_content(model=model, contents=user, config=gen_config)
                except self._errors.APIError as e:
                    failures.append(f"{model}: {e.code}")
                    if e.code in (401, 403) or (e.code == 400 and "API key" in str(e)):
                        raise LLMUnavailable(f"Gemini key rejected ({e.code})") from e
                    if e.code not in self.RETRYABLE:
                        break  # e.g. 404 model retired -> next model
                    if e.code == 429 and "quota" in str(e).lower():
                        # Free-tier daily/minute quota: retrying now cannot succeed; rest the model a while.
                        self._exhausted[model] = time.monotonic() + self.QUOTA_REST_S
                        break
                except Exception as e:  # network errors (connection reset, timeouts)
                    failures.append(f"{model}: {type(e).__name__}")
                else:
                    finish = r.candidates[0].finish_reason.name if r.candidates and r.candidates[0].finish_reason else None
                    if finish not in (None, "STOP"):
                        failures.append(f"{model}: finish_reason={finish}")
                        break
                    usage = r.usage_metadata
                    return parse_json_text(r.text), {
                        "model": model, "input_tokens": getattr(usage, "prompt_token_count", None),
                        "output_tokens": getattr(usage, "candidates_token_count", None), "failures": failures}
                if attempt < len(self.BACKOFF_S):
                    time.sleep(self.BACKOFF_S[attempt])
        raise LLMUnavailable("all Gemini models failed: " + "; ".join(failures))


class GroqProvider(CachedProvider):
    """Groq's OpenAI-compatible chat API with strict JSON-schema output; no extra SDK needed."""
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"
    RETRYABLE = {429, 500, 502, 503, 504}
    TIMEOUT_S = 45
    MAX_WAIT_S = 5

    def __init__(self, models: list[str] | None = None, cache: ResponseCache | None = None):
        super().__init__(cache)
        import requests
        self._requests = requests
        self.key = os.getenv("GROQ_API_KEY")
        if not self.key:
            raise LLMUnavailable("GROQ_API_KEY not set")
        env = os.getenv("GROQ_MODELS")
        self.models = models or ([m.strip() for m in env.split(",") if m.strip()] if env else None) \
            or llm_settings().get("groq_models") or ["openai/gpt-oss-120b"]
        self.model = self.models[0]
        self.temperature = llm_settings().get("temperature")

    def cache_identity(self) -> list:
        return [self.name, self.models[0]]

    def _call(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        failures = []
        for model in self.models:
            body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "response", "schema": schema, "strict": True}}}
            if self.temperature is not None:
                body["temperature"] = self.temperature
            try:
                r = self._requests.post(self.URL, json=body, timeout=self.TIMEOUT_S,
                                        headers={"Authorization": f"Bearer {self.key}"})
                # Free-tier tokens-per-minute limits clear within seconds: wait once rather than downgrade.
                wait = float(r.headers.get("retry-after") or 0) if r.status_code == 429 else 0
                if 0 < wait <= self.MAX_WAIT_S:
                    time.sleep(wait)
                    r = self._requests.post(self.URL, json=body, timeout=self.TIMEOUT_S,
                                            headers={"Authorization": f"Bearer {self.key}"})
            except (self._requests.RequestException, ValueError) as e:
                failures.append(f"{model}: {type(e).__name__}")
                continue
            if r.status_code in (401, 403):
                raise LLMUnavailable(f"Groq key rejected ({r.status_code})")
            if not r.ok:
                failures.append(f"{model}: {r.status_code}")
                continue  # retryable or model-specific (e.g. 400 unsupported schema): next model
            data = r.json()
            choice = data["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                failures.append(f"{model}: finish_reason={choice.get('finish_reason')}")
                continue
            usage = data.get("usage") or {}
            return parse_json_text(choice["message"].get("content")), {
                "model": model, "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"), "failures": failures}
        raise LLMUnavailable("all Groq models failed: " + "; ".join(failures))


class FallbackProvider:
    """Try providers in order; each keeps its own cache, so committed responses still replay first.
    name/model/last_meta describe whichever provider served the latest request (for provenance)."""

    def __init__(self, providers: list):
        self.providers = providers
        self.name, self.model = providers[0].name, providers[0].model
        self.last_meta: dict = {}

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        errors = []
        for provider in self.providers:
            try:
                output = provider.complete_json(system, user, schema)
            except LLMUnavailable as e:
                errors.append(f"{provider.name}: {e}")
                continue
            self.name, self.model = provider.name, provider.model
            self.last_meta = {**provider.last_meta, "fallback_errors": errors}
            return output
        raise LLMUnavailable(" | ".join(errors))


class AnthropicProvider(CachedProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str | None = None, cache: ResponseCache | None = None):
        super().__init__(cache)
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable("anthropic package not installed") from e
        self._anthropic = anthropic
        try:
            # Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile.
            self.client = anthropic.Anthropic()
        except anthropic.AnthropicError as e:
            raise LLMUnavailable(f"no Anthropic credentials: {e}") from e
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.effort = effort or os.getenv("LLM_EFFORT", "low")

    def cache_identity(self) -> list:
        return [self.name, self.model, self.effort]

    def _call(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        anthropic = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                # If the primary model declines, the API re-runs the request on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                anthropic.BadRequestError) as e:
            raise LLMUnavailable(f"{type(e).__name__}: {e}") from e
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # The SDK has already retried transient errors (max_retries=2).
            raise LLMUnavailable(f"{type(e).__name__}: {e}") from e
        except TypeError as e:
            # Missing credentials surface as a TypeError at request time; anything else is a real bug.
            if "authentication" not in str(e).lower():
                raise
            raise LLMUnavailable("no Anthropic credentials (set ANTHROPIC_API_KEY)") from e

        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMUnavailable(f"stop_reason={response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        return parse_json_text(text), {"model": response.model, "input_tokens": response.usage.input_tokens,
                                       "output_tokens": response.usage.output_tokens}


def get_provider():
    name = (os.getenv("LLM_PROVIDER") or llm_settings().get("provider") or "auto").lower()
    if name == "offline":
        return OfflineProvider()
    if name == "auto":
        name = ("gemini" if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                else "groq" if os.getenv("GROQ_API_KEY")
                else "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "offline")
        if name == "offline":
            log.warning("No LLM API key found; using offline templates")
            return OfflineProvider()
    if name == "groq":
        try:
            return GroqProvider()
        except LLMUnavailable as e:
            log.warning("Groq unavailable (%s); using offline templates", e)
            return OfflineProvider()
    try:
        primary = GeminiProvider() if name == "gemini" else AnthropicProvider()
    except LLMUnavailable as e:
        log.warning("LLM provider unavailable (%s); replaying cache with template fallback", e)
        if name == 'gemini':
            return CacheOnlyProvider('gemini', default_gemini_models()[0])
        return CacheOnlyProvider('anthropic', os.getenv('LLM_MODEL', DEFAULT_ANTHROPIC_MODEL),
                                 os.getenv('LLM_EFFORT', 'low'))
    if os.getenv("GROQ_API_KEY"):
        try:
            return FallbackProvider([primary, GroqProvider()])
        except LLMUnavailable as e:
            log.warning("Groq fallback unavailable (%s)", e)
    return primary
