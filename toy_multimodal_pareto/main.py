"""Run the synthetic multimodal Pareto routing demo."""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx

MPL_CONFIG_DIR = Path("outputs") / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / ".cache"))

from cost_models import MetroCostConfig, RoadCostConfig, WalkingCostConfig
from diversity_selection import DiversePathSelector
from graph_builder import GraphConfig, SyntheticGraphBuilder
from multiobjective_routing import BoundsConfig, MultiObjectiveRouter
from pareto_utils import PathResult
from visualization import save_selected_path_visualizations


# Configurable experiment parameters.
GRID_SIZE = 5
SOURCE = 0
DESTINATION = GRID_SIZE * GRID_SIZE - 1
TOP_K = 6
ENABLE_EXTRA_PLOTS = os.getenv("IASP_ENABLE_EXTRA_PLOTS", "0") == "1"
ENABLE_PATH_VISUALIZATIONS = os.getenv("IASP_ENABLE_PATH_VISUALIZATIONS", "1") == "1"
OUTPUT_DIR = Path("outputs")

GRAPH_CONFIG = GraphConfig(
    grid_size=GRID_SIZE,
    spacing_km=1.0,
    walking_radius_km=1.05,
    random_seed=11,
    metro_rows=(1, 3),
    metro_cols=(2,),
    road=RoadCostConfig(
        speed_kmph=28.0,
        mileage_kmpl=15.0,
        fuel_price_per_liter=100.0,
        time_value_per_minute=2.0,
        congestion_cost_weight=1.2,
    ),
    metro=MetroCostConfig(speed_kmph=50.0, base_fare=10.0, fare_per_km=2.5),
    walking=WalkingCostConfig(speed_kmph=5.0),
)

BOUNDS_CONFIG = BoundsConfig(
    alpha_time=8.0,
    alpha_walk=2.5,
    alpha_cost=3.0,
    max_transfers=4,
    min_walk_bound=8.0,
    min_cost_bound=260.0,
)


def main() -> None:
    graph = SyntheticGraphBuilder(GRAPH_CONFIG).build()
    router = MultiObjectiveRouter(graph, BOUNDS_CONFIG)
    pareto_paths = sorted(router.find_pareto_paths(SOURCE, DESTINATION), key=lambda path: path.costs)
    selected_paths = DiversePathSelector(TOP_K).select(pareto_paths)

    bounds = router.compute_bounds(SOURCE, DESTINATION)
    print_network_summary(graph, bounds.optima)
    print(f"Total Pareto paths found: {len(pareto_paths)}\n")

    print("All Pareto paths:")
    for index, path in enumerate(pareto_paths, start=1):
        print_path(index, path)

    print(f"\nFinal Top-{TOP_K} selected paths:")
    for index, path in enumerate(selected_paths, start=1):
        print_path(index, path)

    if ENABLE_PATH_VISUALIZATIONS:
        OUTPUT_DIR.mkdir(exist_ok=True)
        saved_paths = save_selected_path_visualizations(
            graph,
            selected_paths,
            OUTPUT_DIR,
            source=SOURCE,
            destination=DESTINATION,
        )
        print("\nPath visualizations saved:")
        for output_path in saved_paths:
            print(f"  {output_path.resolve()}")

    if ENABLE_EXTRA_PLOTS:
        plot_pareto_front(pareto_paths, selected_paths, OUTPUT_DIR / "pareto_front_time_cost.png")
        plot_graph(graph, OUTPUT_DIR / "synthetic_multimodal_graph.png")
        print(f"\nExtra plots saved to: {OUTPUT_DIR.resolve()}")


def print_network_summary(graph: nx.MultiDiGraph, optima: dict[str, float]) -> None:
    mode_counts: dict[str, int] = {}
    for _, _, edge_data in graph.edges(data=True):
        mode_counts[edge_data["mode"]] = mode_counts.get(edge_data["mode"], 0) + 1

    print("Synthetic multimodal network")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}, Modes: {mode_counts}")
    print(
        "Single-objective optima: "
        f"time={optima['time']:.2f}, walk={optima['walk']:.2f}, "
        f"transfers={optima['transfers']:.0f}, cost={optima['cost']:.2f}\n"
    )


def print_path(index: int, path: PathResult) -> None:
    time, walk, transfers, cost = path.costs
    print(f"{index}. Path: {path.nodes}")
    print(f"   Modes: {path.modes}")
    print(f"   Routes: {path.route_ids}")
    print(f"   Cost: (time={time:.2f}, walk={walk:.2f}, transfers={transfers}, cost={cost:.2f})")


def plot_pareto_front(paths: list[PathResult], selected: list[PathResult], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.scatter([p.costs[0] for p in paths], [p.costs[3] for p in paths], label="Pareto paths", alpha=0.75)
    plt.scatter(
        [p.costs[0] for p in selected],
        [p.costs[3] for p in selected],
        label="Top-K selected",
        marker="x",
        s=90,
    )
    plt.xlabel("Time (minutes)")
    plt.ylabel("Cost")
    plt.title("Pareto front: time vs cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_graph(graph: nx.MultiDiGraph, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from visualization import COLOR_MAP

    pos = nx.get_node_attributes(graph, "pos")

    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=110, node_color="#f6f6f6", edgecolors="#222222")
    nx.draw_networkx_labels(graph, pos, font_size=7)

    for mode, color in COLOR_MAP.items():
        edges = [(u, v) for u, v, data in graph.edges(data=True) if data["mode"] == mode]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edges,
            edge_color=color,
            width=2.4 if mode == "metro" else 1.0,
            alpha=0.65 if mode == "walk" else 0.8,
            arrows=False,
        )

    plt.title("Synthetic multimodal graph")
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color=color, lw=3, label=mode)
            for mode, color in COLOR_MAP.items()
        ],
        loc="upper left",
        frameon=True,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
