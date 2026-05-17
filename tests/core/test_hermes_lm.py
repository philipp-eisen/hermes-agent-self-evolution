"""Tests for Hermes-native LM routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import copy

import dspy

from evolution.core.hermes_lm import HermesCodexLM, create_lm


def test_create_lm_uses_standard_dspy_lm_for_non_hermes_model():
    lm = create_lm("openai/gpt-4.1-mini")

    assert isinstance(lm, dspy.LM)
    assert not isinstance(lm, HermesCodexLM)
    assert lm.model == "openai/gpt-4.1-mini"


def test_create_lm_uses_hermes_codex_lm_for_hermes_codex_prefix():
    lm = create_lm("hermes-codex/gpt-5.5")

    assert isinstance(lm, HermesCodexLM)
    assert lm.hermes_model == "gpt-5.5"


def test_hermes_codex_lm_routes_chat_completion_through_hermes_codex_client():
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="codex auth works"))],
        usage={},
        model="gpt-5.5",
        _hidden_params={},
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch(
        "evolution.core.hermes_lm.HermesCodexLM._build_client",
        return_value=(fake_client, "gpt-5.5"),
    ) as build_client:
        lm = create_lm("hermes-codex/gpt-5.5")
        outputs = lm(messages=[{"role": "user", "content": "ping"}], max_tokens=8)

    build_client.assert_called_once_with(force_refresh=False)
    fake_client.chat.completions.create.assert_called_once()
    request = fake_client.chat.completions.create.call_args.kwargs
    assert request["model"] == "gpt-5.5"
    assert request["messages"] == [{"role": "user", "content": "ping"}]
    assert request["max_tokens"] == 8
    assert outputs == ["codex auth works"]


def test_hermes_codex_lm_refreshes_and_retries_after_auth_failure():
    stale_client = MagicMock()
    stale_client.chat.completions.create.side_effect = RuntimeError("token_invalidated")
    fresh_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="refreshed"))],
        usage={},
        model="gpt-5.5",
        _hidden_params={},
    )
    fresh_client = MagicMock()
    fresh_client.chat.completions.create.return_value = fresh_response

    with patch(
        "evolution.core.hermes_lm.HermesCodexLM._build_client",
        side_effect=[(stale_client, "gpt-5.5"), (fresh_client, "gpt-5.5")],
    ) as build_client:
        lm = create_lm("hermes-codex/gpt-5.5")
        outputs = lm(messages=[{"role": "user", "content": "ping"}], max_tokens=8)

    assert [call.kwargs for call in build_client.call_args_list] == [
        {"force_refresh": False},
        {"force_refresh": True},
    ]
    stale_client.chat.completions.create.assert_called_once()
    fresh_client.chat.completions.create.assert_called_once()
    assert outputs == ["refreshed"]


def test_hermes_codex_lm_deepcopy_drops_live_client():
    lm = create_lm("hermes-codex/gpt-5.5")
    live_client = MagicMock()
    lm._hermes_client = live_client
    lm._resolved_model = "gpt-5.5"

    copied = copy.deepcopy(lm)

    assert isinstance(copied, HermesCodexLM)
    assert copied.hermes_model == "gpt-5.5"
    assert copied._hermes_client is None
    assert copied._resolved_model is None
    assert lm._hermes_client is live_client
