from dataclasses import replace
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from phoenix.db import models
from phoenix.db.insertion.span import insert_span
from phoenix.server.types import DbSessionFactory
from phoenix.trace.schemas import Span, SpanContext, SpanKind, SpanStatusCode

_START_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _span(
    name: str,
    trace_id: str,
    span_id: str,
    *,
    parent_id: str | None = None,
    status_code: SpanStatusCode = SpanStatusCode.OK,
) -> Span:
    return Span(
        name=name,
        context=SpanContext(trace_id=trace_id, span_id=span_id),
        span_kind=SpanKind.CHAIN,
        parent_id=parent_id,
        start_time=_START_TIME,
        end_time=_START_TIME + timedelta(seconds=1),
        status_code=status_code,
        status_message="",
        attributes={},
        events=[],
        conversation=None,
    )


async def test_cumulative_rollups_only_follow_edges_within_the_trace(
    db: DbSessionFactory,
) -> None:
    async with db() as session:
        await insert_span(
            session,
            _span(
                "foreign-child",
                "foreign-child-trace",
                "foreign-child",
                parent_id="target-parent",
                status_code=SpanStatusCode.ERROR,
            ),
            "project",
        )
        await insert_span(
            session,
            _span("target-parent", "target-trace", "target-parent"),
            "project",
        )
        await insert_span(
            session,
            _span("foreign-parent", "foreign-parent-trace", "foreign-parent"),
            "project",
        )
        await insert_span(
            session,
            _span(
                "target-orphan",
                "target-orphan-trace",
                "target-orphan",
                parent_id="foreign-parent",
                status_code=SpanStatusCode.ERROR,
            ),
            "project",
        )

        spans = {
            span.name: span
            for span in await session.scalars(
                select(models.Span).where(models.Span.name.in_(("target-parent", "foreign-parent")))
            )
        }

        assert spans["target-parent"].cumulative_error_count == 0
        assert spans["foreign-parent"].cumulative_error_count == 0


async def test_decision_spans_store_token_counts_from_decision_attributes(
    db: DbSessionFactory,
) -> None:
    span = replace(
        _span("decision", "decision-trace", "decision"),
        span_kind=SpanKind.DECISION,
        attributes={"decision": {"token_count": {"input": 345, "output": 38}}},
    )
    async with db() as session:
        await insert_span(session, span, "project")
        stored = await session.scalar(select(models.Span).where(models.Span.name == "decision"))
    assert stored is not None
    assert stored.llm_token_count_prompt == 345
    assert stored.llm_token_count_completion == 38
    assert stored.cumulative_llm_token_count_prompt == 345
    assert stored.cumulative_llm_token_count_completion == 38
