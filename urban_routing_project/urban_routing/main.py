"""
main.py — CLI entry point for Urban Multimodal Pareto-Optimal Routing.

Usage
-----
  # Full pipeline on real BMTC data
  python main.py --origin "Majestic" --destination "Whitefield" --top-k 5

  # On synthetic data (no Kaggle download required)
  python main.py --synthetic --origin 0 --destination 99 --top-k 5

  # Run benchmark on multiple random OD pairs
  python main.py --synthetic --benchmark --n-pairs 10 --top-k 5

  # Skip plots
  python main.py --synthetic --origin 0 --destination 50 --no-plots
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Ensure project root on path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from data.loader import BMTCLoader
from core.graph import MultimodalGraph
from algorithms.pareto_dijkstra import ParetoDijkstra
from baselines.weighted_sum import WeightedSumRouter
from baselines.lexicographic import LexicographicRouter
from selection.diversity_selector import DiversitySelector
from selection.cluster_selector import ClusterSelector
from evaluation.benchmark import run_benchmark
from evaluation.metrics import summarize_paths
from visualization.plot import (
    plot_pareto_front, plot_parallel_coordinates,
    plot_method_comparison, plot_routes_on_map, plot_radar,
)

console = Console()


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Urban Multimodal Pareto Routing")
    p.add_argument("--origin",       type=str, default=None,
                   help="Origin stop ID or name")
    p.add_argument("--destination",  type=str, default=None,
                   help="Destination stop ID or name")
    p.add_argument("--top-k",        type=int, default=cfg.DEFAULT_TOP_K,
                   help="Number of Top-K routes to select")
    p.add_argument("--synthetic",    action="store_true",
                   help="Force use of synthetic dataset")
    p.add_argument("--benchmark",    action="store_true",
                   help="Run benchmark over multiple OD pairs")
    p.add_argument("--n-pairs",      type=int, default=5,
                   help="Number of random OD pairs for benchmark")
    p.add_argument("--no-plots",     action="store_true",
                   help="Skip generating plots")
    p.add_argument("--weights",      type=float, nargs=5,
                   default=list(cfg.DEFAULT_WEIGHTS),
                   metavar=("W_TIME","W_COST","W_XFER","W_WALK","W_CO2"),
                   help="Weighted-sum weights (must sum to 1)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_stop(graph: MultimodalGraph, query: str) -> Optional[str]:
    """Find a stop_id by exact ID or by fuzzy name match."""
    if graph.has_node(query):
        return query
    query_lower = query.lower()
    for sid, stop in graph.stops.items():
        if query_lower in stop.name.lower():
            return sid
    return None


def random_od_pairs(stop_ids: List[str], n: int, seed: int = 0) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        o, d = rng.sample(stop_ids, 2)
        if o != d:
            pairs.append((o, d))
    return pairs


def print_route_table(paths, title: str = "Top-K Routes"):
    table = Table(title=title, show_lines=True)
    table.add_column("#",          width=3)
    table.add_column("Time (min)", justify="right")
    table.add_column("Cost (₹)",   justify="right")
    table.add_column("Transfers",  justify="right")
    table.add_column("Walk (m)",   justify="right")
    table.add_column("CO₂ (g)",    justify="right")
    table.add_column("Hops",       justify="right")

    for i, path in enumerate(paths, 1):
        w = path.total_weight
        table.add_row(
            str(i),
            f"{w.time_min:.1f}",
            f"{w.cost_inr:.1f}",
            str(int(w.transfers)),
            f"{w.walking_m:.0f}",
            f"{w.co2_g:.0f}",
            str(len(path.nodes) - 1),
        )
    console.print(table)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold blue]Urban Multimodal Pareto-Optimal Routing[/bold blue]\n"
        "BMTC Bengaluru Bus Network — Multi-criteria Dijkstra",
        border_style="blue",
    ))

    # ── Load data ──
    loader = BMTCLoader()
    if args.synthetic:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader._generate_synthetic()
            loader._loaded = True
    else:
        loader.load()

    stops  = loader.get_stops()
    routes = loader.get_routes()

    # ── Build graph ──
    console.print("\n[cyan]Building multi-layer graph...[/cyan]")
    graph = MultimodalGraph()
    graph.build(stops, routes)
    console.print(f"  Nodes: {graph.node_count()}  |  Edges: {graph.edge_count()}")

    stop_ids = list(stops.keys())

    # ── Benchmark mode ──
    if args.benchmark:
        console.print(f"\n[cyan]Running benchmark on {args.n_pairs} random OD pairs...[/cyan]")
        od_pairs = random_od_pairs(stop_ids, args.n_pairs)
        results  = run_benchmark(graph, od_pairs, k=args.top_k, verbose=True)

        if not args.no_plots:
            plot_method_comparison(results)

        # Aggregate summary
        console.print("\n[bold]Aggregate method statistics:[/bold]")
        methods = list(dict.fromkeys(r["method"] for r in results))
        agg_table = Table(show_lines=True)
        agg_table.add_column("Method")
        agg_table.add_column("Avg Runtime (s)", justify="right")
        agg_table.add_column("Avg Pareto Size", justify="right")
        agg_table.add_column("Avg Diversity",   justify="right")
        agg_table.add_column("Avg Hypervolume", justify="right")
        import numpy as np
        for method in methods:
            rows = [r for r in results if r["method"] == method]
            agg_table.add_row(
                method,
                f"{np.mean([r['runtime_s'] for r in rows]):.4f}",
                f"{np.mean([r['pareto_size'] for r in rows]):.1f}",
                f"{np.mean([r.get('diversity_score', 0) for r in rows]):.3f}",
                f"{np.mean([r.get('hypervolume', 0) for r in rows]):.2e}",
            )
        console.print(agg_table)
        return

    # ── Single OD query ──
    origin_query = args.origin or stop_ids[0]
    dest_query   = args.destination or stop_ids[min(30, len(stop_ids) - 1)]

    origin = resolve_stop(graph, str(origin_query))
    dest   = resolve_stop(graph, str(dest_query))

    if origin is None:
        console.print(f"[red]Origin stop '{origin_query}' not found.[/red]")
        sys.exit(1)
    if dest is None:
        console.print(f"[red]Destination stop '{dest_query}' not found.[/red]")
        sys.exit(1)

    console.print(f"\n[green]Origin:[/green]      {graph.stop_name(origin)} ({origin})")
    console.print(f"[green]Destination:[/green] {graph.stop_name(dest)} ({dest})")
    console.print(f"[green]Top-K:[/green]       {args.top_k}\n")

    # ── Pareto Dijkstra ──
    console.print("[cyan]Running Multi-criteria Pareto Dijkstra...[/cyan]")
    router = ParetoDijkstra(graph)
    stats  = router.run_with_stats(origin, dest)
    pareto_paths = stats["paths"]

    console.print(
        f"  Pareto frontier: [bold]{len(pareto_paths)}[/bold] non-dominated paths  "
        f"[dim](runtime: {stats['runtime_s']:.3f}s)[/dim]"
    )

    if not pareto_paths:
        console.print("[red]No paths found between origin and destination.[/red]")
        sys.exit(0)

    # ── Top-K Selection ──
    div_sel    = DiversitySelector(k=args.top_k)
    clust_sel  = ClusterSelector(k=args.top_k)
    div_paths  = div_sel.select(pareto_paths)
    clust_paths= clust_sel.select(pareto_paths)

    console.print("\n[bold yellow]── Diversity-Constrained Selection (Jaccard) ──[/bold yellow]")
    print_route_table(div_paths, title="Top-K: Diversity Selector")

    console.print("\n[bold green]── Cluster-Based Selection (k-means) ──[/bold green]")
    print_route_table(clust_paths, title="Top-K: Cluster Selector")

    # ── Baselines ──
    ws_router  = WeightedSumRouter(graph, weights=tuple(args.weights))
    lex_router = LexicographicRouter(graph)

    ws_result  = ws_router.run(origin, dest)
    lex_result = lex_router.run(origin, dest)

    console.print("\n[bold red]── Baseline: Weighted Sum ──[/bold red]")
    if ws_result:
        print_route_table([ws_result], title="Weighted Sum (single route)")
    else:
        console.print("  [dim]No route found.[/dim]")

    console.print("\n[bold magenta]── Baseline: Lexicographic ──[/bold magenta]")
    if lex_result:
        print_route_table([lex_result], title="Lexicographic (single route)")
    else:
        console.print("  [dim]No route found.[/dim]")

    # ── Metrics comparison ──
    console.print("\n[bold]Metric Summary:[/bold]")
    m_table = Table(show_lines=True)
    m_table.add_column("Method")
    m_table.add_column("Paths",      justify="right")
    m_table.add_column("Diversity",  justify="right")
    m_table.add_column("Hypervolume",justify="right")
    m_table.add_column("Spread(t)",  justify="right")

    for label, paths in [
        ("Pareto+Diversity", div_paths),
        ("Pareto+Cluster",   clust_paths),
        ("WeightedSum",      [ws_result] if ws_result else []),
        ("Lexicographic",    [lex_result] if lex_result else []),
    ]:
        m = summarize_paths(paths)
        m_table.add_row(
            label,
            str(m["n_paths"]),
            f"{m['diversity_score']:.3f}",
            f"{m['hypervolume']:.2e}",
            f"{m['spread_time']:.3f}",
        )
    console.print(m_table)

    # ── Visualizations ──
    if not args.no_plots:
        console.print("\n[cyan]Generating plots → outputs/[/cyan]")
        plot_pareto_front(pareto_paths, selected=div_paths)
        plot_parallel_coordinates(div_paths,
                                  labels=[f"R{i+1}" for i in range(len(div_paths))])
        plot_radar(div_paths, labels=[f"R{i+1}" for i in range(len(div_paths))])
        plot_routes_on_map(div_paths, graph.stops,
                           labels=[f"R{i+1}" for i in range(len(div_paths))])

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
