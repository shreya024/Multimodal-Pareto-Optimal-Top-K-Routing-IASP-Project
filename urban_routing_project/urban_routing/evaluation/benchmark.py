"""
evaluation/benchmark.py — Runtime and quality comparison across all methods.

Compares:
  1. Multi-criteria Pareto Dijkstra + DiversitySelector
  2. Multi-criteria Pareto Dijkstra + ClusterSelector
  3. Weighted-sum scalarization (baseline)
  4. Lexicographic ordering (baseline)

For each (origin, destination) pair the harness collects:
  - wall-clock runtime
  - frontier / result size
  - hypervolume, spread, diversity metrics
  - per-objective best/worst coverage
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from algorithms.pareto_dijkstra import ParetoDijkstra
from baselines.lexicographic import LexicographicRouter
from baselines.weighted_sum import WeightedSumRouter
from core.graph import MultimodalGraph
from data.schema import ParetoPath
from evaluation.metrics import summarize_paths
from selection.cluster_selector import ClusterSelector
from selection.diversity_selector import DiversitySelector
import config as cfg

console = Console()


@dataclass
class MethodResult:
    method:        str
    runtime_s:     float
    all_paths:     List[ParetoPath]       # full Pareto set (or single baseline path)
    selected:      List[ParetoPath]       # Top-K selection
    metrics:       Dict                   = field(default_factory=dict)


def run_benchmark(
    graph:       MultimodalGraph,
    od_pairs:    List[Tuple[str, str]],
    k:           int = cfg.DEFAULT_TOP_K,
    verbose:     bool = True,
) -> List[Dict]:
    """
    Run all four methods on every (origin, destination) pair.

    Returns
    -------
    List of result dicts, one per (od_pair × method).
    """
    pd_router  = ParetoDijkstra(graph)
    ws_router  = WeightedSumRouter(graph)
    lex_router = LexicographicRouter(graph)
    div_sel    = DiversitySelector(k=k)
    clust_sel  = ClusterSelector(k=k)

    all_results = []

    for origin, destination in od_pairs:
        if verbose:
            console.rule(f"[bold cyan]{graph.stop_name(origin)} → {graph.stop_name(destination)}")

        # ── Pareto Dijkstra ──
        t0     = time.perf_counter()
        pareto = pd_router.run(origin, destination)
        pareto_time = time.perf_counter() - t0

        # ── Selection variants ──
        t1       = time.perf_counter()
        div_sel_paths = div_sel.select(pareto)
        div_sel_time  = time.perf_counter() - t1

        t2         = time.perf_counter()
        clust_paths = clust_sel.select(pareto)
        clust_time  = time.perf_counter() - t2

        # ── Weighted sum ──
        ws_stats   = ws_router.run_with_stats(origin, destination)
        # ── Lexicographic ──
        lex_stats  = lex_router.run_with_stats(origin, destination)

        results = [
            MethodResult(
                method="Pareto+Diversity",
                runtime_s=pareto_time + div_sel_time,
                all_paths=pareto,
                selected=div_sel_paths,
                metrics=summarize_paths(div_sel_paths),
            ),
            MethodResult(
                method="Pareto+Cluster",
                runtime_s=pareto_time + clust_time,
                all_paths=pareto,
                selected=clust_paths,
                metrics=summarize_paths(clust_paths),
            ),
            MethodResult(
                method="WeightedSum",
                runtime_s=ws_stats["runtime_s"],
                all_paths=ws_stats["paths"],
                selected=ws_stats["paths"],
                metrics=summarize_paths(ws_stats["paths"]),
            ),
            MethodResult(
                method="Lexicographic",
                runtime_s=lex_stats["runtime_s"],
                all_paths=lex_stats["paths"],
                selected=lex_stats["paths"],
                metrics=summarize_paths(lex_stats["paths"]),
            ),
        ]

        if verbose:
            _print_comparison_table(results, origin, destination, graph)

        for r in results:
            all_results.append({
                "origin":      origin,
                "destination": destination,
                "method":      r.method,
                "runtime_s":   r.runtime_s,
                "pareto_size": len(r.all_paths),
                "selected_k":  len(r.selected),
                **r.metrics,
            })

    return all_results


def _print_comparison_table(
    results:     List[MethodResult],
    origin:      str,
    destination: str,
    graph:       MultimodalGraph,
):
    table = Table(title=f"Method Comparison", show_lines=True)
    table.add_column("Method",         style="bold")
    table.add_column("Runtime (s)",    justify="right")
    table.add_column("Pareto Size",    justify="right")
    table.add_column("Selected K",     justify="right")
    table.add_column("Hypervolume",    justify="right")
    table.add_column("Diversity",      justify="right")
    table.add_column("Spread (time)",  justify="right")

    for r in results:
        m = r.metrics
        table.add_row(
            r.method,
            f"{r.runtime_s:.4f}",
            str(len(r.all_paths)),
            str(len(r.selected)),
            f"{m.get('hypervolume', 0):.2e}",
            f"{m.get('diversity_score', 0):.3f}",
            f"{m.get('spread_time', 0):.3f}",
        )
    console.print(table)

    # Print best path per method
    for r in results:
        if r.selected:
            best = min(r.selected, key=lambda p: p.total_weight.time_min)
            console.print(f"  [yellow]{r.method}[/yellow] best: {best.summary()}")
