"""Stage 3: embed problems locally, cluster them incrementally, name clusters in batches.

Incremental and ID-stable across runs:
1. new problems are embedded once (per embedding model);
2. each unassigned problem joins the closest existing cluster if cosine >= assign_threshold;
3. the remaining pool is clustered with HDBSCAN once it is big enough (noise waits for later);
4. clusters whose centroids are >= merge_threshold similar are merged (the larger survives,
   the smaller is kept as ``merged`` with ``merged_into`` so links never break).
Naming is the only LLM part: 8 clusters per call, only for new or substantially grown clusters.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import HDBSCAN
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from idea_radar.clock import utc_now
from idea_radar.config import Settings
from idea_radar.db.models import Cluster, ClusterMember, ClusterStatus, Problem, ProblemEmbedding
from idea_radar.embed import Embedder, Matrix, from_blob, normalize, to_blob
from idea_radar.llm.client import LLMClient
from idea_radar.llm.errors import LLMOutputError, StopRun
from idea_radar.llm.prompts import load_prompt
from idea_radar.llm.types import LLMRequest
from idea_radar.log import get_logger

_log = get_logger(component="cluster")

NAMING_STAGE = "name_clusters"


def problem_text(who: str, problem: str) -> str:
    return f"{who}: {problem}"


def embed_new_problems(factory: sessionmaker[Session], embedder: Embedder) -> int:
    with factory() as session:
        done = select(ProblemEmbedding.problem_id).where(ProblemEmbedding.model == embedder.name)
        rows = list(session.scalars(select(Problem).where(Problem.id.not_in(done))))
        if not rows:
            return 0
        vectors = embedder.embed([problem_text(p.who, p.problem) for p in rows])
        for problem, vector in zip(rows, vectors, strict=True):
            session.add(
                ProblemEmbedding(problem_id=problem.id, model=embedder.name, vector=to_blob(vector))
            )
        session.commit()
        return len(rows)


@dataclass
class ClusterReport:
    embedded: int = 0
    assigned_existing: int = 0
    new_clusters: int = 0
    clustered_new: int = 0
    merged: int = 0
    noise: int = 0
    active_clusters: int = 0


def _centroid(vectors: list[Matrix]) -> Matrix:
    centroid: Matrix = normalize(np.mean(np.vstack(vectors), axis=0, keepdims=True))[0]
    return centroid


def run_clustering(
    settings: Settings, factory: sessionmaker[Session], embedder: Embedder
) -> ClusterReport:
    cfg = settings.clustering
    report = ClusterReport(embedded=embed_new_problems(factory, embedder))
    now = utc_now()
    with factory() as session:
        vectors = {
            e.problem_id: from_blob(e.vector)
            for e in session.scalars(
                select(ProblemEmbedding).where(ProblemEmbedding.model == embedder.name)
            )
        }
        membership = {m.problem_id: m for m in session.scalars(select(ClusterMember))}
        clusters = {
            c.id: c
            for c in session.scalars(
                select(Cluster).where(Cluster.status == ClusterStatus.ACTIVE.value)
            )
        }
        unassigned = [pid for pid in vectors if pid not in membership]

        # 1) join existing clusters
        if clusters and unassigned:
            ids = list(clusters)
            centroids = np.vstack([from_blob(clusters[c].centroid) for c in ids])
            still: list[int] = []
            for pid in unassigned:
                sims = centroids @ vectors[pid]
                best = int(np.argmax(sims))
                if float(sims[best]) >= cfg.assign_threshold:
                    member = ClusterMember(
                        problem_id=pid, cluster_id=ids[best], similarity=float(sims[best])
                    )
                    session.add(member)
                    membership[pid] = member
                    report.assigned_existing += 1
                else:
                    still.append(pid)
            unassigned = still

        # 2) HDBSCAN on the remaining pool
        if len(unassigned) >= max(cfg.min_pool, cfg.min_cluster_size + 1):
            matrix = np.vstack([vectors[pid] for pid in unassigned])
            labels = HDBSCAN(
                min_cluster_size=cfg.min_cluster_size, metric="euclidean", copy=True
            ).fit_predict(matrix)
            groups: dict[int, list[int]] = defaultdict(list)
            for pid, label in zip(unassigned, labels, strict=True):
                if label >= 0:
                    groups[int(label)].append(pid)
            for label, pids in list(groups.items()):
                # purity guard: HDBSCAN on small data lets outliers ride along
                centroid = _centroid([vectors[p] for p in pids])
                kept = [
                    p for p in pids if float(vectors[p] @ centroid) >= cfg.min_member_similarity
                ]
                groups[label] = kept if len(kept) >= cfg.min_cluster_size else []
            for pids in groups.values():
                if not pids:
                    continue
                cluster = Cluster(
                    centroid=to_blob(_centroid([vectors[p] for p in pids])), size=len(pids)
                )
                session.add(cluster)
                session.flush()
                clusters[cluster.id] = cluster
                for pid in pids:
                    member = ClusterMember(problem_id=pid, cluster_id=cluster.id, similarity=0.0)
                    session.add(member)
                    membership[pid] = member
                report.new_clusters += 1
                report.clustered_new += len(pids)
            report.noise = len(unassigned) - report.clustered_new
        else:
            report.noise = len(unassigned)

        # 3) recompute centroids, sizes, similarities
        def refresh(cluster_ids: list[int]) -> None:
            by_cluster: dict[int, list[int]] = defaultdict(list)
            for pid, m in membership.items():
                if m.cluster_id in cluster_ids and pid in vectors:
                    by_cluster[m.cluster_id].append(pid)
            for cid in cluster_ids:
                pids = by_cluster.get(cid, [])
                if not pids:
                    continue
                centroid = _centroid([vectors[p] for p in pids])
                clusters[cid].centroid = to_blob(centroid)
                clusters[cid].size = len(pids)
                clusters[cid].updated_at = now
                for p in pids:
                    membership[p].similarity = round(float(vectors[p] @ centroid), 4)

        refresh(list(clusters))

        # 4) merge near-duplicate clusters (greedy, largest first)
        merged_any = True
        while merged_any and len(clusters) > 1:
            merged_any = False
            ids = sorted(clusters, key=lambda c: clusters[c].size, reverse=True)
            centroids = np.vstack([from_blob(clusters[c].centroid) for c in ids])
            sims = centroids @ centroids.T
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if float(sims[i, j]) >= cfg.merge_threshold:
                        keep, drop = clusters[ids[i]], clusters[ids[j]]
                        for m in membership.values():
                            if m.cluster_id == drop.id:
                                m.cluster_id = keep.id
                        drop.status = ClusterStatus.MERGED.value
                        drop.merged_into = keep.id
                        drop.size = 0
                        del clusters[drop.id]
                        refresh([keep.id])
                        report.merged += 1
                        merged_any = True
                        break
                if merged_any:
                    break

        session.commit()
        report.active_clusters = len(clusters)
    return report


# ---------------------------------------------------------------- naming (LLM)


class NamedCluster(BaseModel):
    k: int
    name: str
    summary: str
    who: str


class NamingBatch(BaseModel):
    clusters: list[NamedCluster]


@dataclass(frozen=True)
class ClusterExamples:
    cluster_id: int
    size: int
    examples: list[str]


def clusters_needing_names(
    factory: sessionmaker[Session], *, growth: float, per_cluster: int
) -> list[ClusterExamples]:
    with factory() as session:
        clusters = list(
            session.scalars(select(Cluster).where(Cluster.status == ClusterStatus.ACTIVE.value))
        )
        out = []
        for c in clusters:
            if c.name is not None and c.size <= c.named_size * (1 + growth):
                continue
            rows = session.execute(
                select(Problem.who, Problem.problem)
                .join(ClusterMember, ClusterMember.problem_id == Problem.id)
                .where(ClusterMember.cluster_id == c.id)
                .order_by(ClusterMember.similarity.desc())
                .limit(per_cluster)
            ).all()
            out.append(
                ClusterExamples(c.id, c.size, [problem_text(who, prob) for who, prob in rows])
            )
        return out


def render_clusters(groups: list[ClusterExamples]) -> str:
    lines = ["Groups:"]
    for k, group in enumerate(groups):
        lines.append(f"[{k}] ({group.size} problems)")
        lines.extend(f"- {example}" for example in group.examples)
    return "\n".join(lines)


@dataclass
class NamingReport:
    needing: int = 0
    named: int = 0
    failed: int = 0
    stop: StopRun | None = None


async def name_clusters(
    settings: Settings, factory: sessionmaker[Session], client: LLMClient
) -> NamingReport:
    cfg = settings.clustering
    prompt = load_prompt("name_clusters", 1)
    pending = clusters_needing_names(
        factory, growth=cfg.rename_growth, per_cluster=cfg.examples_per_cluster
    )
    report = NamingReport(needing=len(pending))
    for start in range(0, len(pending), cfg.naming_batch):
        batch = pending[start : start + cfg.naming_batch]
        request: LLMRequest[NamingBatch] = LLMRequest(
            stage=NAMING_STAGE,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            system=prompt.text,
            prompt=render_clusters(batch),
            schema=NamingBatch,
            tier="small",
            n_items=len(batch),
        )
        try:
            result = await client.complete_json(request)
        except StopRun as exc:
            report.stop = exc
            break
        except LLMOutputError as exc:
            _log.error("naming_batch_failed", size=len(batch), error=str(exc))
            report.failed += len(batch)
            continue
        named: dict[int, NamedCluster] = {}
        for item in result.data.clusters:
            if 0 <= item.k < len(batch):
                named.setdefault(batch[item.k].cluster_id, item)
        with factory() as session:
            for cluster_id, item in named.items():
                cluster = session.get(Cluster, cluster_id)
                if cluster is None:
                    continue
                cluster.name = item.name.strip()
                cluster.summary = item.summary.strip()
                cluster.who = item.who.strip()
                cluster.named_size = cluster.size
                cluster.named_at = utc_now()
            session.commit()
        report.named += len(named)
        report.failed += len(batch) - len(named)
    return report


def cluster_members_text(factory: sessionmaker[Session], cluster_id: int) -> list[dict[str, Any]]:
    """Members of a cluster with their evidence (for reports and the dashboard)."""
    from idea_radar.db.models import RawItem

    with factory() as session:
        rows = session.execute(
            select(Problem, RawItem, ClusterMember.similarity)
            .join(ClusterMember, ClusterMember.problem_id == Problem.id)
            .join(RawItem, RawItem.id == Problem.raw_item_id)
            .where(ClusterMember.cluster_id == cluster_id)
            .order_by(ClusterMember.similarity.desc())
        ).all()
        return [
            {
                "who": p.who,
                "problem": p.problem,
                "quote": p.quote,
                "wtp": p.wtp,
                "urgency": p.urgency,
                "source": r.source,
                "url": r.url,
                "context": r.context,
                "created_at": r.created_at.isoformat(),
                "similarity": sim,
            }
            for p, r, sim in rows
        ]


def reset_clusters(factory: sessionmaker[Session]) -> int:
    """Delete clusters, memberships and demand scores (embeddings and problems are kept)."""
    from sqlalchemy import delete, update

    from idea_radar.db.models import DemandScore

    with factory() as session:
        n = len(list(session.scalars(select(Cluster.id))))
        session.execute(delete(DemandScore))
        session.execute(delete(ClusterMember))
        session.execute(update(Cluster).values(merged_into=None))
        session.execute(delete(Cluster))
        session.commit()
    return n
