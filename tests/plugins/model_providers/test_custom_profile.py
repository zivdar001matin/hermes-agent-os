"""Unit tests for the custom provider profile's reasoning wiring."""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


_NONE_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
_NONE_TOP = {"reasoning_effort": "none"}


class TestCustomReasoningWireShape:
    def test_no_reasoning_config_uses_none(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5.2"
        )
        assert eb == _NONE_EXTRA
        assert tl == _NONE_TOP

    def test_disabled_uses_none(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert eb == _NONE_EXTRA
        assert tl == _NONE_TOP

    def test_effort_none_uses_none(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="glm-5.2"
        )
        assert eb == _NONE_EXTRA
        assert tl == _NONE_TOP

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_enabled_effort_goes_top_level(self, custom_profile, effort):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": effort}
        assert eb == {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": effort,
            }
        }

    def test_enabled_without_effort_uses_none(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model="glm-5.2"
        )
        assert eb == _NONE_EXTRA
        assert tl == _NONE_TOP


class TestCustomReasoningWithNumCtx:
    def test_num_ctx_alone(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="qwen3"
        )
        assert eb == {"options": {"num_ctx": 8192}, **_NONE_EXTRA}
        assert tl == _NONE_TOP

    def test_num_ctx_with_effort(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            ollama_num_ctx=8192,
            model="qwen3",
        )
        assert eb == {
            "options": {"num_ctx": 8192},
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "high",
            },
        }
        assert tl == {"reasoning_effort": "high"}
