"""
OpenInference ``DECISION`` span conventions.

Decision models (TypeSafe System One, OpenAI Decisions) answer typed questions
rather than generating text, so their spans record ``decision.*`` attributes
instead of ``llm.*``. The ``openinference-semconv`` release Phoenix pins
predates these names, so they are spelled out here, and the helpers below let
token accounting and cost tracking treat an LLM span and a DECISION span the
same way: one model call with a model name, a provider, and input and output
token counts.
"""

from collections.abc import Mapping
from typing import Any, Optional

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from phoenix.trace.attributes import get_attribute_value

DECISION_SPAN_KIND = "DECISION"


class DecisionAttributes:
    SYSTEM = "decision.system"
    PROVIDER = "decision.provider"
    MODEL_NAME = "decision.model_name"
    REQUEST_MODEL_NAME = "decision.request.model_name"
    RESPONSE_MODEL_NAME = "decision.response.model_name"
    TOKEN_COUNT_INPUT = "decision.token_count.input"
    TOKEN_COUNT_OUTPUT = "decision.token_count.output"


def is_model_call_span_kind(span_kind: Optional[str]) -> bool:
    """Whether spans of this kind record a model call with token counts.

    Exact match on the canonical upper-case kind, as elsewhere in Phoenix.
    """
    return span_kind in (OpenInferenceSpanKindValues.LLM.value, DECISION_SPAN_KIND)


def _as_token_count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(value))


def get_model_call_token_counts(
    span_kind: Optional[str],
    attributes: Optional[Mapping[str, Any]],
) -> tuple[Optional[int], Optional[int]]:
    """Input and output token counts for an LLM or DECISION span, else ``(None, None)``.

    LLM spans count ``llm.token_count.prompt`` and ``llm.token_count.completion``;
    DECISION spans count ``decision.token_count.input`` and
    ``decision.token_count.output``. Missing or malformed counts are ``None``.
    """
    kind = (span_kind or "").upper()
    if kind == OpenInferenceSpanKindValues.LLM.value:
        return (
            _as_token_count(get_attribute_value(attributes, SpanAttributes.LLM_TOKEN_COUNT_PROMPT)),
            _as_token_count(
                get_attribute_value(attributes, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION)
            ),
        )
    if kind == DECISION_SPAN_KIND:
        return (
            _as_token_count(get_attribute_value(attributes, DecisionAttributes.TOKEN_COUNT_INPUT)),
            _as_token_count(get_attribute_value(attributes, DecisionAttributes.TOKEN_COUNT_OUTPUT)),
        )
    return None, None


def get_model_call_model_name(attributes: Optional[Mapping[str, Any]]) -> str:
    """``llm.model_name``, else ``decision.model_name``, else an empty string."""
    for key in (SpanAttributes.LLM_MODEL_NAME, DecisionAttributes.MODEL_NAME):
        value = get_attribute_value(attributes, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_model_call_provider(attributes: Optional[Mapping[str, Any]]) -> str:
    """``llm.provider``, else ``decision.provider``, else an empty string."""
    for key in (SpanAttributes.LLM_PROVIDER, DecisionAttributes.PROVIDER):
        value = get_attribute_value(attributes, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
