from __future__ import annotations

import pytest

from idea_radar.llm.guard import (
    ApiCredentialInEnvError,
    api_credentials_in_env,
    ensure_subscription_auth,
)


def test_no_credentials_passes() -> None:
    ensure_subscription_auth(backend="agent_sdk", allow_api_key_env=False, environ={})


@pytest.mark.parametrize("var", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_credential_blocks_agent_sdk(var: str) -> None:
    with pytest.raises(ApiCredentialInEnvError, match=var):
        ensure_subscription_auth(
            backend="agent_sdk", allow_api_key_env=False, environ={var: "sk-test"}
        )


def test_explicit_opt_in_allows_it() -> None:
    ensure_subscription_auth(
        backend="agent_sdk", allow_api_key_env=True, environ={"ANTHROPIC_API_KEY": "sk-test"}
    )


def test_api_backend_is_not_blocked() -> None:
    ensure_subscription_auth(
        backend="anthropic_api", allow_api_key_env=False, environ={"ANTHROPIC_API_KEY": "sk"}
    )


def test_empty_values_are_ignored() -> None:
    assert api_credentials_in_env({"ANTHROPIC_API_KEY": "  "}) == []
