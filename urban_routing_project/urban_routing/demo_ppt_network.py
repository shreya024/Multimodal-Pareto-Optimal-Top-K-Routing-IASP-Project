"""
Report demo: 5x5 multimodal network matching the presentation story.

This file is intentionally separate from the real-data pipeline. It builds a
small, deterministic network with:
  - a 5x5 road grid,
  - one horizontal and one vertical metro line,
  - walking/access links between nearby road and metro nodes.

It then runs the existing routing stack: ParetoDijkstra, diversity Top-K,
cluster Top-K, weighted-sum baseline, and lexicographic baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx

from algorithms.pareto_dijkstra import ParetoDijkstra
from baselines.lexicographic import LexicographicRouter
from baselines.weighted_sum import WeightedSumRouter
from core.graph import MultimodalGraph
from data.schema import EdgeWeight, GraphEdge, ParetoPath, Route, Stop, TransportMode
from evaluation.metrics import summarize_paths
from selection.cluster_selector import ClusterSelector
from selection.diversity_selector import DiversitySelector


GRID_N = 5
SPACING_M = 500.0
ORIGIN = "R_0_0"
DESTINATION = "R_4_4"
TOP_K = 5


def _latlon(x: int, y: int) -> Tuple[float, float]:
    return 12.95 + y * 0.0045, 77.55 + x * 0.0045


def _weight(mode: str, distance_m: float = SPACING_M) -> EdgeWeight:
    if mode == "road":
        return EdgeWeight(
            time_min=6.0,
            cost_inr=8.0,
            transfers=0.0,
            walking_m=0.0,
            co2_g=450.0,
        )
    if mode == "metro":
        return EdgeWeight(
            time_min=3.0,
            cost_inr=4.0,
            transfers=0.0,
            walking_m=0.0,
            co2_g=100.0,
        )
    if mode == "walk":
        return EdgeWeight(
            time_min=12.0 * (distance_m / SPACING_M),
            cost_inr=0.0,
            transfers=0.0,
            walking_m=distance_m,
            co2_g=0.0,
        )
    if mode == "transfer":
        return EdgeWeight(
            time_min=3.0,
            cost_inr=0.0,
            transfers=1.0,
            walking_m=120.0,
            co2_g=0.0,
        )
    raise ValueError(mode)


def _add_edge(graph: MultimodalGraph, edge: GraphEdge) -> None:
    key = graph.G.add_edge(edge.src, edge.dst, edge=edge, weight=edge.weight.time_min)
    graph.edge_data[(edge.src, edge.dst, key)] = edge


def _add_bidirectional(
    graph: MultimodalGraph,
    src: str,
    dst: str,
    route_id: str | None,
    mode: TransportMode,
    weight: EdgeWeight,
    distance_m: float,
) -> None:
    for u, v in ((src, dst), (dst, src)):
        _add_edge(
            graph,
            GraphEdge(
                src=u,
                dst=v,
                route_id=route_id,
                mode=mode,
                weight=weight,
                distance_m=distance_m,
            ),
        )


def build_demo_graph() -> MultimodalGraph:
    graph = MultimodalGraph()
    graph.G = nx.MultiDiGraph()

    stops: Dict[str, Stop] = {}
    routes: Dict[str, Route] = {}

    for y in range(GRID_N):
        for x in range(GRID_N):
            sid = f"R_{x}_{y}"
            lat, lon = _latlon(x, y)
            stops[sid] = Stop(
                stop_id=sid,
                name=f"Road ({x},{y})",
                lat=lat,
                lon=lon,
                mode=TransportMode.BUS,
            )

    for x in range(GRID_N):
        sid = f"M_H_{x}"
        lat, lon = _latlon(x, 2)
        stops[sid] = Stop(sid, f"Metro H{x}", lat, lon, TransportMode.METRO)

    for y in range(GRID_N):
        sid = f"M_V_{y}"
        lat, lon = _latlon(2, y)
        stops[sid] = Stop(sid, f"Metro V{y}", lat, lon, TransportMode.METRO)

    graph.stops = stops
    graph.routes = routes
    for sid, stop in stops.items():
        graph.G.add_node(sid, stop=stop)

    road_route_stops = [stops[f"R_{x}_{y}"] for y in range(GRID_N) for x in range(GRID_N)]
    routes["road:grid"] = Route("road:grid", "5x5 road grid", TransportMode.BUS, road_route_stops)

    road_w = _weight("road")
    walk_w = _weight("walk")
    for y in range(GRID_N):
        for x in range(GRID_N):
            if x + 1 < GRID_N:
                _add_bidirectional(graph, f"R_{x}_{y}", f"R_{x+1}_{y}", "road:grid", TransportMode.BUS, road_w, SPACING_M)
                _add_bidirectional(graph, f"R_{x}_{y}", f"R_{x+1}_{y}", None, TransportMode.WALK, walk_w, SPACING_M)
            if y + 1 < GRID_N:
                _add_bidirectional(graph, f"R_{x}_{y}", f"R_{x}_{y+1}", "road:grid", TransportMode.BUS, road_w, SPACING_M)
                _add_bidirectional(graph, f"R_{x}_{y}", f"R_{x}_{y+1}", None, TransportMode.WALK, walk_w, SPACING_M)

    metro_h = [stops[f"M_H_{x}"] for x in range(GRID_N)]
    metro_v = [stops[f"M_V_{y}"] for y in range(GRID_N)]
    routes["metro:horizontal"] = Route("metro:horizontal", "straight east-west metro", TransportMode.METRO, metro_h)
    routes["metro:vertical"] = Route("metro:vertical", "straight north-south metro", TransportMode.METRO, metro_v)

    metro_w = _weight("metro")
    for x in range(GRID_N - 1):
        _add_bidirectional(graph, f"M_H_{x}", f"M_H_{x+1}", "metro:horizontal", TransportMode.METRO, metro_w, SPACING_M)
    for y in range(GRID_N - 1):
        _add_bidirectional(graph, f"M_V_{y}", f"M_V_{y+1}", "metro:vertical", TransportMode.METRO, metro_w, SPACING_M)

    transfer_w = _weight("transfer")
    for x in range(GRID_N):
        _add_bidirectional(graph, f"R_{x}_2", f"M_H_{x}", None, TransportMode.TRANSFER, transfer_w, 120.0)
    for y in range(GRID_N):
        _add_bidirectional(graph, f"R_2_{y}", f"M_V_{y}", None, TransportMode.TRANSFER, transfer_w, 120.0)
    _add_bidirectional(graph, "M_H_2", "M_V_2", "metro:interchange", TransportMode.TRANSFER, transfer_w, 120.0)

    return graph


def _path_modes(path: ParetoPath) -> str:
    modes = []
    for edge in path.edges:
        if edge.mode.value not in modes:
            modes.append(edge.mode.value)
    return " + ".join(modes)


def _row(label: str, path: ParetoPath) -> str:
    w = path.total_weight
    return (
        f"| {label} | {w.time_min:.1f} | {w.walking_m:.0f} | "
        f"{int(w.transfers)} | {w.cost_inr:.1f} | {w.co2_g:.0f} | "
        f"{len(path.edges)} | {_path_modes(path)} |"
    )


def _table(title: str, paths: Iterable[ParetoPath]) -> List[str]:
    lines = [
        f"## {title}",
        "",
        "| Route | Time | Walk | Transfers | Cost | CO2 | Hops | Modes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, path in enumerate(paths, 1):
        lines.append(_row(f"R{idx}", path))
    lines.append("")
    return lines


def write_report(
    graph: MultimodalGraph,
    pareto: List[ParetoPath],
    diversity: List[ParetoPath],
    cluster: List[ParetoPath],
    weighted: ParetoPath | None,
    lexicographic: ParetoPath | None,
) -> Path:
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "demo_ppt_network_report.md"

    lines = [
        "# 5x5 Multimodal Pareto Routing Demo",
        "",
        "This miniature network mirrors the presentation narrative: a grid road network, straight metro lines, and walking/access links between nearby nodes.",
        "",
        f"- Origin: `{ORIGIN}`",
        f"- Destination: `{DESTINATION}`",
        f"- Nodes: `{graph.node_count()}`",
        f"- Edges: `{graph.edge_count()}`",
        f"- Pareto frontier size: `{len(pareto)}`",
        "",
        "The demo is deliberately small so the report can explain why scalar shortest path is insufficient: the fastest, cheapest, lowest-walk, and lowest-transfer paths are not the same route.",
        "",
    ]
    lines.extend(_table("Diversity-Constrained Top-K", diversity))
    lines.extend(_table("Cluster Top-K", cluster))
    if weighted:
        lines.extend(_table("Weighted-Sum Baseline", [weighted]))
    if lexicographic:
        lines.extend(_table("Lexicographic Baseline", [lexicographic]))

    for label, selected in (
        ("Diversity Top-K", diversity),
        ("Cluster Top-K", cluster),
        ("Weighted Sum", [weighted] if weighted else []),
        ("Lexicographic", [lexicographic] if lexicographic else []),
    ):
        metrics = summarize_paths(selected)
        lines.extend([
            f"## Metrics: {label}",
            "",
            f"- Paths: `{metrics['n_paths']}`",
            f"- Diversity score: `{metrics['diversity_score']:.3f}`",
            f"- Hypervolume: `{metrics['hypervolume']:.2e}`",
            f"- Time spread: `{metrics['spread_time']:.3f}`",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    graph = build_demo_graph()
    pareto = ParetoDijkstra(graph).run(ORIGIN, DESTINATION)
    diversity = DiversitySelector(k=TOP_K).select(pareto)
    cluster = ClusterSelector(k=TOP_K).select(pareto)
    weighted = WeightedSumRouter(graph).run(ORIGIN, DESTINATION)
    lexicographic = LexicographicRouter(graph).run(ORIGIN, DESTINATION)
    report_path = write_report(graph, pareto, diversity, cluster, weighted, lexicographic)

    print("5x5 PPT demo complete")
    print(f"Nodes: {graph.node_count()}, edges: {graph.edge_count()}")
    print(f"Pareto frontier: {len(pareto)} non-dominated paths")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
