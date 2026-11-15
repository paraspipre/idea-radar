"""Cluster-quality eval: labeled problem pairs (same underlying problem?) + unsupervised stats."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import silhouette_score
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.clock import utc_now
from idea_radar.db.models import Cluster, ClusterMember, ClusterStatus, Problem, ProblemEmbedding
from idea_radar.embed import from_blob
from idea_radar.evals.metrics import Binary

PAIRS_FILE = Path("evals") / "data" / "cluster_pairs.jsonl"


class PairLabel(BaseModel):
    a: str  # raw_item_id of problem A (stable across DB rebuilds of the same items)
    b: str
    a_text: str
    b_text: str
    same: bool
    labeled_at: str


@dataclass(frozen=True)
class PairSide:
    raw_item_id: str
    text: str


def load_pairs(path: Path) -> list[PairLabel]:
    if not path.is_file():
        return []
    return [
        PairLabel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_pair(path: Path, a: PairSide, b: PairSide, *, same: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = PairLabel(
        a=a.raw_item_id,
        b=b.raw_item_id,
        a_text=a.text,
        b_text=b.text,
        same=same,
        labeled_at=utc_now().isoformat(),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label.model_dump(), ensure_ascii=False) + "\n")


def _assignments(
    factory: sessionmaker[Session], model: str
) -> tuple[dict[str, int], dict[str, np.ndarray[Any, Any]], dict[str, str]]:
    """raw_item_id -> active cluster id, raw_item_id -> vector, raw_item_id -> text."""
    with factory() as session:
        status = {c.id: (c.status, c.merged_into) for c in session.scalars(select(Cluster))}

        def resolve(cid: int) -> int:
            seen = set()
            while status.get(cid, (ClusterStatus.ACTIVE.value, None))[0] != "active":
                nxt = status[cid][1]
                if nxt is None or cid in seen:
                    break
                seen.add(cid)
                cid = nxt
            return cid

        cluster_of = {
            pid: resolve(cid)
            for pid, cid in session.execute(
                select(ClusterMember.problem_id, ClusterMember.cluster_id)
            ).all()
        }
        vectors: dict[str, np.ndarray[Any, Any]] = {}
        texts: dict[str, str] = {}
        assigned: dict[str, int] = {}
        for problem, emb in session.execute(
            select(Problem, ProblemEmbedding)
            .join(ProblemEmbedding, ProblemEmbedding.problem_id == Problem.id)
            .where(ProblemEmbedding.model == model)
        ).all():
            key = problem.raw_item_id
            vectors[key] = from_blob(emb.vector)
            texts[key] = f"{problem.who}: {problem.problem}"
            if problem.id in cluster_of:
                assigned[key] = cluster_of[problem.id]
    return assigned, vectors, texts


def candidate_pairs(
    factory: sessionmaker[Session],
    *,
    model: str,
    exclude: set[tuple[str, str]],
    n: int,
    seed: int | None = None,
) -> list[tuple[PairSide, PairSide]]:
    """Half same-cluster pairs, half nearest neighbours in different clusters (hard negatives)."""
    rng = random.Random(seed)
    assigned, vectors, texts = _assignments(factory, model)
    keys = list(vectors)
    same: list[tuple[str, str]] = []
    by_cluster: dict[int, list[str]] = {}
    for key, cid in assigned.items():
        by_cluster.setdefault(cid, []).append(key)
    for members in by_cluster.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                same.append((members[i], members[j]))
    hard: list[tuple[str, str]] = []
    if len(keys) > 1:
        matrix = np.vstack([vectors[k] for k in keys])
        sims = matrix @ matrix.T
        for i, key in enumerate(keys):
            order = np.argsort(-sims[i])
            for j in order[1:4]:
                other = keys[int(j)]
                if assigned.get(key) is None or assigned.get(key) != assigned.get(other):
                    hard.append((key, other))
                    break
    rng.shuffle(same)
    rng.shuffle(hard)
    picked: list[tuple[str, str]] = []
    seen = set(exclude)
    for a, b in [*same[: n // 2], *hard]:
        pair = tuple(sorted((a, b)))
        if pair in seen:
            continue
        seen.add(pair)  # type: ignore[arg-type]
        picked.append((pair[0], pair[1]))
        if len(picked) >= n:
            break
    rng.shuffle(picked)
    return [(PairSide(a, texts[a]), PairSide(b, texts[b])) for a, b in picked]


def cluster_quality(
    factory: sessionmaker[Session], *, model: str, pairs: list[PairLabel]
) -> dict[str, Any]:
    assigned, vectors, _ = _assignments(factory, model)
    n_problems = len(vectors)
    clustered = [k for k in vectors if k in assigned]
    labels = [assigned[k] for k in clustered]
    silhouette: float | None = None
    if len(set(labels)) >= 2 and len(clustered) > len(set(labels)):
        matrix = np.vstack([vectors[k] for k in clustered])
        silhouette = float(silhouette_score(matrix, labels, metric="cosine"))
    result: dict[str, Any] = {
        "date": utc_now().isoformat(),
        "model": model,
        "n_problems": n_problems,
        "n_clustered": len(clustered),
        "n_clusters": len(set(labels)),
        "noise_ratio": (1 - len(clustered) / n_problems) if n_problems else None,
        "silhouette": silhouette,
    }
    scored = [p for p in pairs if p.a in vectors and p.b in vectors]
    if scored:
        metric = Binary.from_pairs(
            (
                assigned.get(p.a) is not None and assigned.get(p.a) == assigned.get(p.b),
                p.same,
            )
            for p in scored
        )
        result["pairwise"] = metric.as_dict()
    return result


def latest_ranked_clusters(factory: sessionmaker[Session], *, limit: int) -> list[dict[str, Any]]:
    """Active clusters with their most recent demand score, best first."""
    from idea_radar.db.models import DemandScore, RawItem

    with factory() as session:
        latest: dict[int, DemandScore] = {}
        for score in session.scalars(select(DemandScore).order_by(DemandScore.created_at)):
            latest[score.cluster_id] = score
        clusters = {
            c.id: c
            for c in session.scalars(
                select(Cluster).where(Cluster.status == ClusterStatus.ACTIVE.value)
            )
        }
        sources: dict[int, set[str]] = {}
        for cid, source in session.execute(
            select(ClusterMember.cluster_id, RawItem.source)
            .join(Problem, Problem.id == ClusterMember.problem_id)
            .join(RawItem, RawItem.id == Problem.raw_item_id)
        ).all():
            sources.setdefault(cid, set()).add(source)
        rows: list[dict[str, Any]] = [
            {
                "cluster_id": cid,
                "name": clusters[cid].name,
                "summary": clusters[cid].summary,
                "who": clusters[cid].who,
                "size": clusters[cid].size,
                "n_sources": len(sources.get(cid, set())),
                "score": latest[cid].score,
                "components": latest[cid].components,
                "weekly_counts": latest[cid].weekly_counts,
            }
            for cid in clusters
            if cid in latest
        ]
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows[:limit]
