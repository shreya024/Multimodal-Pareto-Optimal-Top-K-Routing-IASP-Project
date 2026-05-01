"""
Generate report-ready figures for both:
  1. the 5x5 presentation demo network, and
  2. the actual BMTC + Metro + walk routing graph.

Images are written under the repository root:
  report_images/demo/
  report_images/actual/
"""
from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from algorithms.pareto_dijkstra import ParetoDijkstra
from baselines.lexicographic import LexicographicRouter
from baselines.weighted_sum import WeightedSumRouter
from core.graph import MultimodalGraph
from data.fuser import MultimodalFuser
from data.schema import ParetoPath, TransportMode
from demo_ppt_network import DESTINATION, ORIGIN, TOP_K, build_demo_graph
from evaluation.metrics import diversity_score, summarize_paths
from selection.cluster_selector import ClusterSelector
from selection.diversity_selector import DiversitySelector


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "report_images" / "demo"
ACTUAL_DIR = ROOT / "report_images" / "actual"

MODE_COLORS = {
    "bus": "#E76F51",
    "metro": "#2A9D8F",
    "walk": "#6C757D",
    "transfer": "#8A5CF6",
}


def _ensure_dirs() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    ACTUAL_DIR.mkdir(parents=True, exist_ok=True)


def _run_stack(graph: MultimodalGraph, origin: str, destination: str, k: int):
    t0 = time.perf_counter()
    pareto = ParetoDijkstra(graph).run(origin, destination)
    pareto_runtime = time.perf_counter() - t0

    diversity = DiversitySelector(k=k).select(pareto)
    cluster = ClusterSelector(k=k).select(pareto)
    weighted = WeightedSumRouter(graph).run(origin, destination)
    lexicographic = LexicographicRouter(graph).run(origin, destination)

    return {
        "pareto": pareto,
        "diversity": diversity,
        "cluster": cluster,
        "weighted": weighted,
        "lexicographic": lexicographic,
        "runtime": pareto_runtime,
    }


def _coords(graph: MultimodalGraph, node: str) -> Tuple[float, float]:
    stop = graph.stops[node]
    return stop.lon, stop.lat


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
        path.write_bytes(buffer.getvalue())
    finally:
        plt.close(fig)
    print(f"saved {path}")


def plot_network_layers(graph: MultimodalGraph, title: str, path: Path, max_edges: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))

    edges = []
    for _, _, _, data in graph.G.edges(keys=True, data=True):
        edge = data.get("edge")
        if edge is not None:
            edges.append(edge)
    if max_edges is not None:
        edges = edges[:max_edges]

    for edge in edges:
        if edge.src not in graph.stops or edge.dst not in graph.stops:
            continue
        x1, y1 = _coords(graph, edge.src)
        x2, y2 = _coords(graph, edge.dst)
        color = MODE_COLORS.get(edge.mode.value, "#999999")
        lw = 1.8 if edge.mode in (TransportMode.BUS, TransportMode.METRO) else 0.8
        alpha = 0.5 if edge.mode in (TransportMode.BUS, TransportMode.METRO) else 0.25
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=1)

    for mode in (TransportMode.BUS, TransportMode.METRO, TransportMode.WALK):
        xs, ys = [], []
        for stop in graph.stops.values():
            if stop.mode == mode:
                xs.append(stop.lon)
                ys.append(stop.lat)
        if xs:
            ax.scatter(
                xs,
                ys,
                s=18 if mode != TransportMode.METRO else 34,
                c=MODE_COLORS[mode.value],
                label=f"{mode.value.title()} nodes",
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Longitude / x")
    ax.set_ylabel("Latitude / y")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.18)
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, path)


def plot_pareto_front(pareto: List[ParetoPath], selected: List[ParetoPath], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    times = [p.total_weight.time_min for p in pareto]
    costs = [p.total_weight.cost_inr for p in pareto]
    transfers = [p.total_weight.transfers for p in pareto]

    sc = ax.scatter(times, costs, c=transfers, cmap="viridis", s=42, alpha=0.75, label="Pareto path")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Transfers")

    st = [p.total_weight.time_min for p in selected]
    scost = [p.total_weight.cost_inr for p in selected]
    ax.scatter(st, scost, marker="*", s=180, color="#E63946", edgecolor="black", linewidth=0.7, label="Diversity Top-K")
    for i, (x, y) in enumerate(zip(st, scost), 1):
        ax.annotate(f"R{i}", (x, y), xytext=(7, 5), textcoords="offset points", fontsize=9)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Travel time (min)")
    ax.set_ylabel("Cost (INR)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_route_overlay(
    graph: MultimodalGraph,
    paths: List[ParetoPath],
    title: str,
    path: Path,
    background: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))

    if background:
        xs = [s.lon for s in graph.stops.values()]
        ys = [s.lat for s in graph.stops.values()]
        ax.scatter(xs, ys, s=7, c="#CBD5E1", alpha=0.75, zorder=0)

    colors = cm.tab10(np.linspace(0, 1, len(paths)))
    for idx, (route, color) in enumerate(zip(paths, colors), 1):
        for edge in route.edges:
            if edge.src not in graph.stops or edge.dst not in graph.stops:
                continue
            x1, y1 = _coords(graph, edge.src)
            x2, y2 = _coords(graph, edge.dst)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.6, alpha=0.85, zorder=2)
        if route.nodes:
            sx, sy = _coords(graph, route.nodes[0])
            dx, dy = _coords(graph, route.nodes[-1])
            ax.scatter([sx], [sy], marker="o", s=80, color=color, edgecolor="black", zorder=3)
            ax.scatter([dx], [dy], marker="s", s=80, color=color, edgecolor="black", zorder=3)
            ax.plot([], [], color=color, linewidth=3, label=f"R{idx}")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Longitude / x")
    ax.set_ylabel("Latitude / y")
    ax.legend(title="Selected routes", loc="best")
    ax.grid(alpha=0.18)
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, path)


def plot_parallel_objectives(paths: List[ParetoPath], title: str, path: Path) -> None:
    labels = ["Time", "Cost", "Transfers", "Walk", "CO2"]
    vecs = np.array([p.total_weight.as_tuple() for p in paths], dtype=float)
    mins = vecs.min(axis=0)
    maxs = vecs.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1.0
    norm = (vecs - mins) / rng

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = cm.tab10(np.linspace(0, 1, len(paths)))
    x = np.arange(len(labels))
    for idx, (row, color) in enumerate(zip(norm, colors), 1):
        ax.plot(x, row, marker="o", linewidth=2.4, color=color, label=f"R{idx}")

    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["Best in set", "Middle", "Worst in set"])
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=min(5, len(paths)), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def plot_method_comparison(results: Dict, title: str, path: Path) -> None:
    methods = {
        "Pareto + Diversity": results["diversity"],
        "Pareto + Cluster": results["cluster"],
        "Weighted Sum": [results["weighted"]] if results["weighted"] else [],
        "Lexicographic": [results["lexicographic"]] if results["lexicographic"] else [],
    }

    names = list(methods)
    counts = [len(methods[name]) for name in names]
    diversities = [diversity_score(methods[name]) for name in names]
    time_ranges = []
    for name in names:
        paths = methods[name]
        if len(paths) < 2:
            time_ranges.append(0.0)
        else:
            vals = [p.total_weight.time_min for p in paths]
            time_ranges.append(max(vals) - min(vals))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    metric_pack = [
        ("Routes returned", counts, "#457B9D"),
        ("Structural diversity", diversities, "#2A9D8F"),
        ("Time range covered", time_ranges, "#E76F51"),
    ]

    for ax, (label, values, color) in zip(axes, metric_pack):
        bars = ax.bar(names, values, color=color, edgecolor="black", linewidth=0.7)
        ax.set_title(label, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_story_board(results: Dict, title: str, path: Path) -> None:
    selected = results["diversity"]
    rows = []
    for idx, route in enumerate(selected, 1):
        w = route.total_weight
        rows.append([
            f"R{idx}",
            f"{w.time_min:.1f}",
            f"{w.cost_inr:.1f}",
            f"{int(w.transfers)}",
            f"{w.walking_m:.0f}",
            f"{w.co2_g:.0f}",
        ])

    fig, ax = plt.subplots(figsize=(8.5, 2.7))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Route", "Time", "Cost", "Transfers", "Walk", "CO2"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1D3557")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if r % 2 else "#E9ECEF")
    ax.set_title(title, fontweight="bold", pad=12)
    _save(fig, path)


def generate_demo() -> None:
    graph = build_demo_graph()
    results = _run_stack(graph, ORIGIN, DESTINATION, TOP_K)

    plot_network_layers(graph, "Demo network: 5x5 road grid + two metro lines + walking links", DEMO_DIR / "01_demo_network_topology.png")
    plot_pareto_front(results["pareto"], results["diversity"], "Demo Pareto frontier: time-cost tradeoff", DEMO_DIR / "02_demo_pareto_front.png")
    plot_route_overlay(graph, results["diversity"], "Demo diversity Top-K route overlay", DEMO_DIR / "03_demo_selected_routes.png")
    plot_parallel_objectives(results["diversity"], "Demo selected routes: normalized objective profiles", DEMO_DIR / "04_demo_objective_profiles.png")
    plot_method_comparison(results, "Demo method comparison", DEMO_DIR / "05_demo_method_comparison.png")
    plot_story_board(results, "Demo Top-K values for report table", DEMO_DIR / "06_demo_topk_table.png")

    _write_caption_file(DEMO_DIR, graph, results, ORIGIN, DESTINATION)


def generate_actual() -> None:
    fuser = MultimodalFuser(use_osm=False).load()
    graph = MultimodalGraph().build(
        stops=fuser.all_stops,
        routes=fuser.all_routes,
        extra_edges=fuser.extra_edges,
        transfer_edges=fuser.transfer_edges,
    )
    origin = graph.find_stop_by_name("Majestic")[0]
    destination = graph.find_stop_by_name("Whitefield")[0]
    results = _run_stack(graph, origin, destination, 5)

    plot_network_layers(graph, "Actual graph layers: BMTC bus + Namma Metro + walking transfers", ACTUAL_DIR / "01_actual_network_layers.png")
    plot_pareto_front(results["pareto"], results["diversity"], "Actual Pareto frontier: Majestic to Whitefield", ACTUAL_DIR / "02_actual_pareto_front.png")
    plot_route_overlay(graph, results["diversity"], "Actual diversity Top-K route overlay", ACTUAL_DIR / "03_actual_selected_routes.png")
    plot_parallel_objectives(results["diversity"], "Actual selected routes: normalized objective profiles", ACTUAL_DIR / "04_actual_objective_profiles.png")
    plot_method_comparison(results, "Actual method comparison", ACTUAL_DIR / "05_actual_method_comparison.png")
    plot_story_board(results, "Actual Top-K values for report table", ACTUAL_DIR / "06_actual_topk_table.png")

    _write_caption_file(ACTUAL_DIR, graph, results, origin, destination)


def _write_caption_file(folder: Path, graph: MultimodalGraph, results: Dict, origin: str, destination: str) -> None:
    metrics = summarize_paths(results["diversity"])
    lines = [
        "# Figure Captions",
        "",
        f"- Origin: `{graph.stop_name(origin)}`",
        f"- Destination: `{graph.stop_name(destination)}`",
        f"- Nodes: `{graph.node_count()}`",
        f"- Edges: `{graph.edge_count()}`",
        f"- Pareto frontier size: `{len(results['pareto'])}`",
        f"- Pareto runtime: `{results['runtime']:.3f}s`",
        f"- Diversity Top-K score: `{metrics['diversity_score']:.3f}`",
        "",
        "Suggested report order:",
        "1. Network topology/layers",
        "2. Pareto frontier",
        "3. Selected routes overlay",
        "4. Objective profiles",
        "5. Method comparison",
        "6. Top-K table",
        "",
    ]
    (folder / "captions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    _ensure_dirs()
    generate_demo()
    generate_actual()


if __name__ == "__main__":
    main()
