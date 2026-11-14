"""Guard against accidentally billing the Anthropic API.

The Agent SDK prefers API credentials from the environment over the Claude subscription
login. If one of these variables is set, calls would be billed to an API account instead of
the user's Pro plan, so the ``agent_sdk`` backend refuses to run.
"""

from __future__ import annotations

from collections.abc import Mapping

from idea_radar.llm.errors import LLMConfigError

CREDENTIAL_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class ApiCredentialInEnvError(LLMConfigError):
    """An API credential is present while the subscription-backed backend is selected."""


def api_credentials_in_env(environ: Mapping[str, str]) -> list[str]:
    """Names of API credential variables that are set to a non-empty value."""
    return [name for name in CREDENTIAL_ENV_VARS if environ.get(name, "").strip()]


def remediation_message(found: list[str]) -> str:
    names = ", ".join(found)
    first = found[0] if found else "ANTHROPIC_API_KEY"
    return (
        f"{names} is set. With the agent_sdk backend this would bill the Anthropic API "
        "instead of your Claude subscription. Remove it from .env and from your environment. "
        f"PowerShell, this session: Remove-Item Env:{first}. "
        f"Permanently: [Environment]::SetEnvironmentVariable('{first}', $null, 'User')."
    )


def ensure_subscription_auth(
    *, backend: str, allow_api_key_env: bool, environ: Mapping[str, str]
) -> None:
    """Raise ``ApiCredentialInEnvError`` if the agent_sdk backend would bill the API."""
    if backend != "agent_sdk" or allow_api_key_env:
        return
    found = api_credentials_in_env(environ)
    if found:
        raise ApiCredentialInEnvError(remediation_message(found))
