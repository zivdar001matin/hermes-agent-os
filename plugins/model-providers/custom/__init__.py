"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config unset / disabled → top-level reasoning_effort="none"
    + extra_body.chat_template_kwargs.reasoning_effort="none" (MLX/Qwen/vLLM)
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

_DEFAULT_TEMPLATE_REASONING_EFFORT = "none"


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — template kwargs and num_ctx support."""

    @staticmethod
    def _resolve_template_reasoning_effort(reasoning_config: dict | None) -> str:
        """Map Hermes reasoning_config to chat_template_kwargs.reasoning_effort."""
        if not reasoning_config or not isinstance(reasoning_config, dict):
            return _DEFAULT_TEMPLATE_REASONING_EFFORT
        effort = (reasoning_config.get("effort") or "").strip().lower()
        enabled = reasoning_config.get("enabled", True)
        if effort == "none" or enabled is False:
            return _DEFAULT_TEMPLATE_REASONING_EFFORT
        if effort:
            return effort
        return _DEFAULT_TEMPLATE_REASONING_EFFORT

    @staticmethod
    def _apply_template_reasoning(
        extra_body: dict[str, Any], top_level: dict[str, Any], effort: str
    ) -> None:
        top_level["reasoning_effort"] = effort
        if effort == _DEFAULT_TEMPLATE_REASONING_EFFORT:
            # llama.cpp / Qwen GGUF expects enable_thinking in chat_template_kwargs.
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": True,
                "reasoning_effort": effort,
            }

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        template_effort = self._resolve_template_reasoning_effort(reasoning_config)
        self._apply_template_reasoning(extra_body, top_level, template_effort)

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    default_max_tokens=65536,
)

register_provider(custom)
