"""DSPy LM helpers for Hermes-native providers.

The self-evolution package normally lets DSPy/LiteLLM own model routing. Hermes'
OpenAI Codex OAuth route is different: the token lives in Hermes' auth store and
requests must pass through Hermes' Codex Responses adapter. This module exposes a
small DSPy-compatible LM wrapper so optimization runs can use the same Codex auth
as the active Hermes agent loop.
"""

from __future__ import annotations

from typing import Any

import dspy
from openai import OpenAI

from agent.auxiliary_client import CodexAuxiliaryClient, _codex_cloudflare_headers
from hermes_cli.auth import resolve_codex_runtime_credentials

HERMES_CODEX_PREFIX = "hermes-codex/"


class HermesCodexLM(dspy.LM):
    """DSPy LM that routes calls through Hermes' OpenAI Codex OAuth adapter."""

    def __init__(self, model: str, **kwargs: Any):
        hermes_model = model.removeprefix(HERMES_CODEX_PREFIX)
        super().__init__(model=f"{HERMES_CODEX_PREFIX}{hermes_model}", **kwargs)
        self.hermes_model = hermes_model
        self._hermes_client = None
        self._resolved_model = None

    def __getstate__(self):
        """Make the LM deepcopy-safe for DSPy optimizers.

        DSPy optimizers such as MIPROv2 deepcopy the configured LM while
        proposing instructions. Hermes' Codex client owns an httpx/OpenAI client
        with thread locks, which cannot be pickled/deep-copied. Drop the live
        client from copied state and lazily rebuild it on the next call.
        """
        state = self.__dict__.copy()
        state["_hermes_client"] = None
        state["_resolved_model"] = None
        return state

    def _build_client(self, *, force_refresh: bool = False):
        runtime = resolve_codex_runtime_credentials(force_refresh=force_refresh)
        token = str(runtime.get("api_key") or "").strip()
        base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
        if not token or not base_url:
            raise RuntimeError(
                "Hermes Codex OAuth credentials are unavailable. Run `hermes auth add openai-codex` "
                "or select the Codex provider with `hermes model`, then retry."
            )
        raw_client = OpenAI(
            api_key=token,
            base_url=base_url,
            default_headers=_codex_cloudflare_headers(token),
        )
        return CodexAuxiliaryClient(raw_client, self.hermes_model), self.hermes_model

    def _client_and_model(self):
        if self._hermes_client is None:
            self._hermes_client, self._resolved_model = self._build_client(force_refresh=False)
        return self._hermes_client, self._resolved_model

    def _refresh_client(self):
        self._hermes_client, self._resolved_model = self._build_client(force_refresh=True)
        return self._hermes_client, self._resolved_model

    @staticmethod
    def _looks_like_auth_failure(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in {401, 403}:
            return True
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "401",
                "403",
                "token_invalidated",
                "invalidated",
                "unauthorized",
                "authentication",
            )
        )

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        request_kwargs = {**self.kwargs, **kwargs}
        cache = request_kwargs.pop("cache", None)
        request_kwargs.pop("rollout_id", None)
        # DSPy passes cache/rollout metadata for LiteLLM; Hermes' OpenAI-style
        # client should only receive API kwargs.
        _ = cache

        request_messages = messages or [{"role": "user", "content": prompt}]

        def call(client, resolved_model):
            response = client.chat.completions.create(
                model=resolved_model,
                messages=request_messages,
                **request_kwargs,
            )
            if not getattr(response, "model", None):
                response.model = resolved_model
            usage = getattr(response, "usage", None)
            if usage is None:
                response.usage = {}
            elif not isinstance(usage, dict):
                try:
                    response.usage = usage.model_dump()
                except AttributeError:
                    response.usage = vars(usage)
            if not getattr(response, "_hidden_params", None):
                response._hidden_params = {}
            return response

        client, resolved_model = self._client_and_model()
        try:
            return call(client, resolved_model)
        except Exception as exc:
            if not self._looks_like_auth_failure(exc):
                raise
            client, resolved_model = self._refresh_client()
            return call(client, resolved_model)


def create_lm(model: str, **kwargs: Any) -> dspy.LM:
    """Create a DSPy LM, including Hermes-native providers.

    Use ``hermes-codex/<model>`` (for example ``hermes-codex/gpt-5.5``) to use
    the Codex OAuth credentials already configured for Hermes Agent.
    """

    if model.startswith(HERMES_CODEX_PREFIX):
        return HermesCodexLM(model, **kwargs)
    return dspy.LM(model, **kwargs)
