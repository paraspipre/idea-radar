"""The pipeline as a LangGraph graph with typed state (stages 2-8).

State holds only counters and flags; all data lives in the database, and every node is
idempotent against it (it only processes what is pending). That makes resume trivial: a run
that stopped on budget or a usage limit is simply run again.

After every LLM-using node a conditional edge checks for a stop and jumps to ``finalize``.
"""

from __future__ import annotations

import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.config import Settings
from idea_radar.embed import Embedder
from idea_radar.llm.client import LLMClient
from idea_radar.llm.errors import StopRun
from idea_radar.log import bind_context, get_logger
from idea_radar.pipeline.cluster import name_clusters, run_clustering
from idea_radar.pipeline.demand import RankedCluster, run_demand
from idea_radar.pipeline.extract import run_extraction

_log = get_logger(component="graph")

STAGES = ("extract", "cluster", "name", "demand")
LLM_STAGES = {"extract", "name", "competition", "sizing", "skeptic"}


class PipelineState(TypedDict, total=False):
    run_id: str
    done: list[str]
    stopped: bool
    summary: dict[str, Any]


@dataclass
class PipelineContext:
    settings: Settings
    factory: sessionmaker[Session]
    client: LLMClient
    embedder: Embedder
    run_id: str
    until: str = ""  # empty = run every registered stage
    skip: frozenset[str] = frozenset()
    stop: StopRun | None = None
    ranked: list[RankedCluster] = field(default_factory=list)
    extra_stages: list[tuple[str, Callable[[PipelineContext], Awaitable[dict[str, Any]]]]] = field(
        default_factory=list
    )


Node = Callable[[PipelineState], Awaitable[PipelineState]]
StageFn = Callable[[PipelineContext], Awaitable[dict[str, Any]]]

# Stages 5-8 register here (pipeline/opportunity.py) so the graph stays one ordered chain.
EXTRA_STAGES: list[tuple[str, StageFn]] = []


def pipeline_stages() -> tuple[str, ...]:
    return STAGES + tuple(name for name, _ in EXTRA_STAGES)


def _wrap(ctx: PipelineContext, name: str, body: Callable[[], Awaitable[dict[str, Any]]]) -> Node:
    async def node(state: PipelineState) -> PipelineState:
        done = [*state.get("done", []), name]
        summary = dict(state.get("summary", {}))
        if name in ctx.skip:
            summary[name] = {"skipped": True}
            return {"done": done, "summary": summary}
        if ctx.stop is not None and name in LLM_STAGES:
            # After a budget/usage stop, LLM stages are skipped but free stages still run,
            # so a stopped run still leaves fresh clusters, scores and rankings behind.
            summary[name] = {"skipped": "stopped"}
            return {"done": done, "summary": summary, "stopped": True}
        bind_context(run_id=ctx.run_id, stage=name)
        summary[name] = await body()
        _log.info("stage_done", **{k: v for k, v in summary[name].items() if k != "top"})
        return {"done": done, "summary": summary, "stopped": ctx.stop is not None}

    return node


def build_graph(ctx: PipelineContext) -> Any:
    settings, factory, client = ctx.settings, ctx.factory, ctx.client

    async def extract() -> dict[str, Any]:
        report = await run_extraction(settings, factory, client, run_id=ctx.run_id)
        ctx.stop = ctx.stop or report.stop
        return {
            "processed": report.processed,
            "problems": report.problems,
            "not_problems": report.not_problems,
            "failed": report.failed,
        }

    async def cluster() -> dict[str, Any]:
        report = run_clustering(settings, factory, ctx.embedder)
        return dict(vars(report))

    async def name() -> dict[str, Any]:
        report = await name_clusters(settings, factory, client)
        ctx.stop = ctx.stop or report.stop
        return {"needing": report.needing, "named": report.named, "failed": report.failed}

    async def demand() -> dict[str, Any]:
        ctx.ranked = run_demand(settings, factory, run_id=ctx.run_id)
        return {"scored": len(ctx.ranked)}

    bodies: list[tuple[str, Callable[[], Awaitable[dict[str, Any]]]]] = [
        ("extract", extract),
        ("cluster", cluster),
        ("name", name),
        ("demand", demand),
    ]
    for stage_name, fn in ctx.extra_stages:

        def bind(
            f: Callable[[PipelineContext], Awaitable[dict[str, Any]]],
        ) -> Callable[[], Awaitable[dict[str, Any]]]:
            async def body() -> dict[str, Any]:
                return await f(ctx)

            return body

        bodies.append((stage_name, bind(fn)))

    names = [n for n, _ in bodies]
    last = names.index(ctx.until) if ctx.until in names else len(names) - 1
    active = bodies[: last + 1]

    graph = StateGraph(PipelineState)
    for stage_name, body in active:
        graph.add_node(stage_name, cast(Any, _wrap(ctx, stage_name, body)))

    async def finalize(state: PipelineState) -> PipelineState:
        return {"stopped": ctx.stop is not None}

    graph.add_node("finalize", finalize)
    graph.add_edge(START, active[0][0])
    for (current, _), (following, _) in itertools.pairwise(active):
        remaining = [n for n, _ in active][[n for n, _ in active].index(following) :]

        def route(
            state: PipelineState, nxt: str = following, rest: tuple[str, ...] = tuple(remaining)
        ) -> str:
            # Stopped and nothing free left to do -> straight to finalize.
            if state.get("stopped") and all(stage in LLM_STAGES for stage in rest):
                return "finalize"
            return nxt

        graph.add_conditional_edges(current, route, {following: following, "finalize": "finalize"})
    graph.add_edge(active[-1][0], "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


async def run_pipeline(ctx: PipelineContext) -> PipelineState:
    graph = build_graph(ctx)
    result: PipelineState = await graph.ainvoke({"run_id": ctx.run_id, "done": [], "summary": {}})
    return result
