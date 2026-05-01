"""
main.py — CLI entry point for Multimodal Pareto-Optimal Routing.
Layers: BMTC bus (synthetic/real CSV) + Namma Metro (real GTFS) + walk.

Usage
-----
  # Synthetic bus + real Metro GTFS
  python main.py --origin "Majestic" --destination "Whitefield" --top-k 5

  # Force synthetic bus data
  python main.py --synthetic --origin "Majestic" --destination "Whitefield"

  # Benchmark
  python main.py --benchmark --n-pairs 8 --top-k 5

  # List all metro stations
  python main.py --list-metro

  # Skip plots
  python main.py --origin "Majestic" --destination "Indiranagar" --no-plots
"""
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from data.fuser import MultimodalFuser
from data.loader import BMTCLoader
from core.graph import MultimodalGraph
from algorithms.pareto_dijkstra import ParetoDijkstra
from baselines.weighted_sum import WeightedSumRouter
from baselines.lexicographic import LexicographicRouter
from selection.diversity_selector import DiversitySelector
from selection.cluster_selector import ClusterSelector
from evaluation.benchmark import run_benchmark
from evaluation.metrics import summarize_paths

console = Console()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Pareto Routing — BMTC + Namma Metro")
    p.add_argument("--origin",       type=str, default=None)
    p.add_argument("--destination",  type=str, default=None)
    p.add_argument("--top-k",        type=int, default=cfg.DEFAULT_TOP_K)
    p.add_argument("--synthetic",    action="store_true",
                   help="Force synthetic bus data (Metro GTFS always used if present)")
    p.add_argument("--benchmark",    action="store_true")
    p.add_argument("--n-pairs",      type=int, default=6)
    p.add_argument("--no-plots",     action="store_true")
    p.add_argument("--list-metro",   action="store_true",
                   help="Print all metro stations and exit")
    p.add_argument("--use-osm",      action="store_true",
                   help="Add OSM pedestrian walk layer (requires internet)")
    p.add_argument("--weights",      type=float, nargs=5,
                   default=list(cfg.DEFAULT_WEIGHTS),
                   metavar=("W_TIME","W_COST","W_XFER","W_WALK","W_CO2"))
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_system(args) -> Tuple[MultimodalGraph, MultimodalFuser]:
    fuser = MultimodalFuser(use_osm=args.use_osm)

    if args.synthetic:
        # Patch bus loader to use synthetic directly
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fuser.bus_loader = BMTCLoader()
            fuser.bus_loader._generate_synthetic()
            fuser.bus_loader._loaded = True
        bus_stops  = fuser.bus_loader.get_stops()
        bus_routes = fuser.bus_loader.get_routes()
        fuser.all_stops.update(bus_stops)
        fuser.all_routes.update(bus_routes)
        print(f"[fuser] Bus   : {len(bus_stops):4d} stops (synthetic), "
              f"{len(bus_routes):3d} routes")

        # Still load real metro
        from data.metro import MetroLoader
        fuser.metro_loader = MetroLoader()
        fuser.metro_loader.load()
        fuser.all_stops.update(fuser.metro_loader.stops)
        fuser.all_routes.update(fuser.metro_loader.routes)
        fuser.extra_edges.extend(fuser.metro_loader.segment_edges)
        fuser.extra_edges.extend(fuser.metro_loader.interchange_edges)
        print(f"[fuser] Metro : {len(fuser.metro_loader.stops):4d} stations, "
              f"{len(fuser.metro_loader.routes):3d} lines")

        n = fuser._build_bus_metro_transfers(bus_stops, fuser.metro_loader.stops)
        print(f"[fuser] Bus↔Metro transfer edges : {n}")
    else:
        fuser.load()

    graph = MultimodalGraph()
    graph.build(
        stops          = fuser.all_stops,
        routes         = fuser.all_routes,
        extra_edges    = fuser.extra_edges,
        transfer_edges = fuser.transfer_edges,
    )
    return graph, fuser


def resolve_stop(graph: MultimodalGraph, query: str) -> Optional[str]:
    """Exact ID match first, then fuzzy name search."""
    if graph.has_node(query):
        return query
    matches = graph.find_stop_by_name(query)
    if matches:
        # Prefer exact match
        for sid in matches:
            if graph.stops[sid].name.lower() == query.lower():
                return sid
        return matches[0]
    return None


def random_connected_pairs(graph: MultimodalGraph, n: int, seed: int = 0) -> List[Tuple[str,str]]:
    """Sample random OD pairs that have at least one path between them."""
    import networkx as nx
    rng = random.Random(seed)
    stop_ids = list(graph.stops.keys())
    pairs = []
    attempts = 0
    while len(pairs) < n and attempts < n * 20:
        o, d = rng.sample(stop_ids, 2)
        if nx.has_path(graph.G, o, d):
            pairs.append((o, d))
        attempts += 1
    return pairs


def print_route_table(paths, title: str = "Routes"):
    if not paths:
        console.print(f"  [dim]No routes found.[/dim]")
        return
    table = Table(title=title, show_lines=True)
    table.add_column("#",             width=3)
    table.add_column("Time (min)",    justify="right")
    table.add_column("Cost (₹)",      justify="right")
    table.add_column("Transfers",     justify="right")
    table.add_column("Walk (m)",      justify="right")
    table.add_column("CO₂ (g)",       justify="right")
    table.add_column("Hops",          justify="right")
    table.add_column("Modes used")
    for i, path in enumerate(paths, 1):
        w = path.total_weight
        modes = sorted({e.mode.value for e in path.edges if e.mode.value != "transfer"})
        table.add_row(
            str(i),
            f"{w.time_min:.1f}",
            f"{w.cost_inr:.1f}",
            str(int(w.transfers)),
            f"{w.walking_m:.0f}",
            f"{w.co2_g:.0f}",
            str(len(path.nodes) - 1),
            " + ".join(modes),
        )
    console.print(table)


def print_route_detail(path, graph: MultimodalGraph, label: str = ""):
    """Print hop-by-hop itinerary with cumulative time and cost."""
    if not path or not path.edges:
        return
    console.print(f"\n  [bold]{label} Itinerary:[/bold]")
    console.print(f"  [dim]{'Stop':40s}  {'Seg time':>9}  {'Seg cost':>9}  {'Total time':>10}  {'Total cost':>10}[/dim]")
    current_mode = None
    cum_min  = 0.0
    cum_cost = 0.0
    for edge in path.edges:
        mode     = edge.mode.value
        src_name = graph.stop_name(edge.src)
        dst_name = graph.stop_name(edge.dst)
        cum_min  += edge.weight.time_min
        cum_cost += edge.weight.cost_inr
        if mode != current_mode:
            mode_tag = {
                "bus":      "[green]🚌 BUS[/green]",
                "metro":    "[blue]🚇 METRO[/blue]",
                "walk":     "[yellow]🚶 WALK[/yellow]",
                "transfer": "[magenta]↔ TRANSFER[/magenta]",
            }.get(mode, mode.upper())
            console.print(f"  {mode_tag}")
            current_mode = mode
        leg = f"{src_name} → {dst_name}"
        console.print(
            f"    [dim]{leg:40s}  "
            f"{edge.weight.time_min:>6.1f} min  "
            f"₹{edge.weight.cost_inr:>7.1f}  "
            f"{cum_min:>7.1f} min  "
            f"₹{cum_cost:>7.1f}[/dim]"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold blue]🚌 + 🚇  Urban Multimodal Pareto-Optimal Routing[/bold blue]\n"
        "BMTC Bus  +  Namma Metro (BMRCL GTFS)  +  Walk",
        border_style="blue",
    ))

    # ── List metro stations ──
    if args.list_metro:
        from data.metro import MetroLoader
        m = MetroLoader().load()
        table = Table(title="Namma Metro Stations", show_lines=True)
        table.add_column("ID"); table.add_column("Name"); table.add_column("Line")
        purple_ids = {s.stop_id for s in m.routes.get("1", type("R",[],{"stops":[]})()).stops} \
                     | {s.stop_id for s in m.routes.get("2", type("R",[],{"stops":[]})()).stops}
        for sid, stop in m.stops.items():
            line = "Purple" if sid in purple_ids else "Green"
            table.add_row(sid, stop.name, line)
        console.print(table)
        return

    # ── Build system ──
    console.print("\n[cyan]Loading transit data...[/cyan]")
    graph, fuser = build_system(args)

    stop_ids = list(graph.stops.keys())

    # ── Benchmark ──
    if args.benchmark:
        console.print(f"\n[cyan]Benchmark: {args.n_pairs} random connected OD pairs...[/cyan]")
        pairs = random_connected_pairs(graph, args.n_pairs)
        if not pairs:
            console.print("[red]Could not find connected OD pairs.[/red]")
            return
        results = run_benchmark(graph, pairs, k=args.top_k, verbose=True)
        if not args.no_plots:
            from visualization.plot import plot_method_comparison
            plot_method_comparison(results)
        return

    # ── Single query ──
    o_query = args.origin      or "Majestic"
    d_query = args.destination or "Whitefield"

    origin = resolve_stop(graph, o_query)
    dest   = resolve_stop(graph, d_query)

    if origin is None:
        console.print(f"[red]Stop not found: '{o_query}'[/red]")
        console.print("Tip: use --list-metro to see metro stations, "
                      "or try partial names like 'MG Road', 'Koramangala'")
        sys.exit(1)
    if dest is None:
        console.print(f"[red]Stop not found: '{d_query}'[/red]")
        sys.exit(1)

    console.print(f"\n  [green]Origin     :[/green] {graph.stop_name(origin)} "
                  f"[dim]({origin}, {graph.stop_mode(origin)})[/dim]")
    console.print(f"  [green]Destination:[/green] {graph.stop_name(dest)} "
                  f"[dim]({dest}, {graph.stop_mode(dest)})[/dim]")
    console.print(f"  [green]Top-K      :[/green] {args.top_k}\n")

    # ── Pareto Dijkstra ──
    console.print("[cyan]Running Multi-criteria Pareto Dijkstra...[/cyan]")
    router = ParetoDijkstra(graph)
    stats  = router.run_with_stats(origin, dest)
    pareto = stats["paths"]

    if not pareto:
        console.print("[red]No paths found. Try different stops or --synthetic.[/red]")
        sys.exit(0)

    console.print(
        f"  Pareto frontier: [bold]{len(pareto)}[/bold] non-dominated paths  "
        f"[dim]({stats['runtime_s']:.3f}s)[/dim]"
    )

    # ── Top-K selection ──
    div_paths   = DiversitySelector(k=args.top_k).select(pareto)
    clust_paths = ClusterSelector(k=args.top_k).select(pareto)

    console.print("\n[bold yellow]── Diversity-Constrained Top-K (Jaccard) ──[/bold yellow]")
    print_route_table(div_paths, "Top-K: Diversity Selector")
    if div_paths:
        best = min(div_paths, key=lambda p: p.total_weight.time_min)
        print_route_detail(best, graph, "Fastest route")

    console.print("\n[bold green]── Cluster-Based Top-K (k-means) ──[/bold green]")
    print_route_table(clust_paths, "Top-K: Cluster Selector")

    # ── Baselines ──
    ws_path  = WeightedSumRouter(graph, tuple(args.weights)).run(origin, dest)
    lex_path = LexicographicRouter(graph).run(origin, dest)

    console.print("\n[bold red]── Baseline: Weighted Sum ──[/bold red]")
    print_route_table([ws_path] if ws_path else [], "Weighted Sum")
    if ws_path:
        print_route_detail(ws_path, graph, "Weighted sum route")

    console.print("\n[bold magenta]── Baseline: Lexicographic ──[/bold magenta]")
    print_route_table([lex_path] if lex_path else [], "Lexicographic")

    # ── Metrics comparison ──
    console.print("\n[bold]Quality Metrics:[/bold]")
    m_table = Table(show_lines=True)
    m_table.add_column("Method",       style="bold")
    m_table.add_column("Paths",        justify="right")
    m_table.add_column("Diversity",    justify="right")
    m_table.add_column("Hypervolume",  justify="right")
    m_table.add_column("Spread(time)", justify="right")
    m_table.add_column("Obj Range — Time / Cost / Transfers")

    for label, paths in [
        ("Pareto+Diversity", div_paths),
        ("Pareto+Cluster",   clust_paths),
        ("WeightedSum",      [ws_path]  if ws_path  else []),
        ("Lexicographic",    [lex_path] if lex_path else []),
    ]:
        m = summarize_paths(paths)
        obj = m.get("objective_range", {})
        m_table.add_row(
            label,
            str(m["n_paths"]),
            f"{m['diversity_score']:.3f}",
            f"{m['hypervolume']:.2e}",
            f"{m['spread_time']:.3f}",
            f"{obj.get('Time (min)',0):.1f}min / "
            f"₹{obj.get('Cost (INR)',0):.1f} / "
            f"{obj.get('Transfers',0):.1f}",
        )
    console.print(m_table)

    # ── Plots ──
    if not args.no_plots:
        console.print("\n[cyan]Generating plots → outputs/[/cyan]")
        from visualization.plot import (
            plot_pareto_front, plot_parallel_coordinates,
            plot_radar, plot_routes_on_map,
        )
        plot_pareto_front(pareto, selected=div_paths,
                          title=f"Pareto Front: {graph.stop_name(origin)} → {graph.stop_name(dest)}")
        plot_parallel_coordinates(div_paths,
                                  labels=[f"R{i+1}" for i in range(len(div_paths))])
        plot_radar(div_paths, labels=[f"R{i+1}" for i in range(len(div_paths))])
        plot_routes_on_map(div_paths, graph.stops,
                           labels=[f"R{i+1}" for i in range(len(div_paths))])
        console.print("  Saved: pareto_front.png, parallel_coords.png, radar.png, route_map.png")

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()