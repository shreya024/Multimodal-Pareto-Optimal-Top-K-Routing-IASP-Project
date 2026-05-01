"""Detailed visualizations for selected multimodal Pareto paths."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / ".cache"))
Path("outputs").mkdir(exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from pareto_utils import PathResult

COLOR_MAP = {
    "walk": "green",
    "road": "red",
    "metro": "blue",
}

PATH_LINE_STYLES = ["solid", "dashed", "dashdot", "dotted"]


def plot_all_paths(
    graph: nx.MultiDiGraph,
    paths: list[PathResult],
    output_path: str | Path = "outputs/all_paths.png",
    *,
    source: int | None = None,
    destination: int | None = None,
) -> None:
    """Plot all selected paths on one graph with mode-colored segments."""

    if not paths:
        raise ValueError("paths must not be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source, destination = _resolve_endpoints(paths, source, destination)
    pos = _positions(graph)

    fig, (ax, text_ax) = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        gridspec_kw={"width_ratios": [3.0, 1.35]},
    )
    _draw_base_graph(graph, pos, ax)

    for index, path in enumerate(paths, start=1):
        line_style = PATH_LINE_STYLES[(index - 1) % len(PATH_LINE_STYLES)]
        alpha = max(0.45, 1.0 - (index - 1) * 0.08)
        _draw_path_segments(
            graph,
            path,
            pos,
            ax,
            linewidth=3.0,
            alpha=alpha,
            line_style=line_style,
            label_prefix=f"Path {index}",
        )

    _highlight_endpoints(graph, pos, ax, source, destination)
    _add_all_paths_legend(ax, len(paths))
    _write_annotations(text_ax, paths)
    ax.set_title("Top-K Pareto paths by transport mode")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_single_path(
    graph: nx.MultiDiGraph,
    path: PathResult,
    index: int,
    output_path: str | Path | None = None,
    *,
    source: int | None = None,
    destination: int | None = None,
) -> None:
    """Plot one selected path with mode-colored segments and its cost vector."""

    if output_path is None:
        output_path = Path("outputs") / f"path_{index}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source, destination = _resolve_endpoints([path], source, destination)
    pos = _positions(graph)

    fig, (ax, text_ax) = plt.subplots(
        1,
        2,
        figsize=(11, 6.5),
        gridspec_kw={"width_ratios": [3.0, 1.2]},
    )
    _draw_base_graph(graph, pos, ax)
    _draw_path_segments(graph, path, pos, ax, linewidth=4.0, alpha=1.0, line_style="solid")
    _highlight_endpoints(graph, pos, ax, source, destination)
    _add_mode_legend(ax)
    _write_annotations(text_ax, [path], start_index=index)
    ax.set_title(f"Path {index}: mode-colored route")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_selected_path_visualizations(
    graph: nx.MultiDiGraph,
    paths: list[PathResult],
    output_dir: str | Path = "outputs",
    *,
    source: int | None = None,
    destination: int | None = None,
) -> list[Path]:
    """Save all required visualizations and return their file paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [output_dir / "all_paths.png"]
    plot_all_paths(graph, paths, saved_paths[0], source=source, destination=destination)

    for index, path in enumerate(paths, start=1):
        output_path = output_dir / f"path_{index}.png"
        plot_single_path(graph, path, index, output_path, source=source, destination=destination)
        saved_paths.append(output_path)

    return saved_paths


def _positions(graph: nx.MultiDiGraph) -> dict[int, tuple[float, float]]:
    pos = nx.get_node_attributes(graph, "pos")
    if len(pos) != graph.number_of_nodes():
        raise ValueError("every graph node must have a 'pos' attribute")
    return pos


def _resolve_endpoints(
    paths: list[PathResult],
    source: int | None,
    destination: int | None,
) -> tuple[int, int]:
    first_path = paths[0]
    if source is None:
        source = first_path.nodes[0]
    if destination is None:
        destination = first_path.nodes[-1]
    return source, destination


def _draw_base_graph(graph: nx.MultiDiGraph, pos: dict[int, tuple[float, float]], ax) -> None:
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="lightgray",
        width=0.5,
        alpha=0.22,
        arrows=False,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=80,
        node_color="#f2f2f2",
        edgecolors="#bdbdbd",
        linewidths=0.6,
    )


def _draw_path_segments(
    graph: nx.MultiDiGraph,
    path: PathResult,
    pos: dict[int, tuple[float, float]],
    ax,
    *,
    linewidth: float,
    alpha: float,
    line_style: str,
    label_prefix: str | None = None,
) -> None:
    for segment_index, (u, v) in enumerate(zip(path.nodes, path.nodes[1:])):
        mode = _segment_mode(graph, path, segment_index, u, v)
        color = COLOR_MAP.get(mode, "black")
        label = label_prefix if segment_index == 0 and label_prefix is not None else None
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=[(u, v)],
            edge_color=color,
            width=linewidth,
            alpha=alpha,
            style=line_style,
            arrows=True,
            arrowsize=14,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.07",
            label=label,
        )


def _segment_mode(graph: nx.MultiDiGraph, path: PathResult, segment_index: int, u: int, v: int) -> str:
    expected_mode = path.modes[segment_index] if segment_index < len(path.modes) else None
    expected_route = path.route_ids[segment_index] if segment_index < len(path.route_ids) else None

    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        raise ValueError(f"path segment ({u}, {v}) is not an edge in the graph")

    for attrs in edge_data.values():
        if attrs.get("mode") == expected_mode and attrs.get("route_id") == expected_route:
            return str(attrs["mode"])
    for attrs in edge_data.values():
        if attrs.get("mode") == expected_mode:
            return str(attrs["mode"])
    return str(next(iter(edge_data.values()))["mode"])


def _highlight_endpoints(
    graph: nx.MultiDiGraph,
    pos: dict[int, tuple[float, float]],
    ax,
    source: int,
    destination: int,
) -> None:
    nx.draw_networkx_nodes(graph, pos, nodelist=[source], node_color="black", node_size=180, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[destination],
        node_color="gold",
        edgecolors="#7a5a00",
        linewidths=1.2,
        node_size=190,
        ax=ax,
    )


def _write_annotations(text_ax, paths: list[PathResult], start_index: int = 1) -> None:
    text_ax.axis("off")
    blocks = [_format_annotation(start_index + offset, path) for offset, path in enumerate(paths)]
    text_ax.text(
        0.0,
        1.0,
        "\n\n".join(blocks),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        linespacing=1.25,
    )


def _format_annotation(index: int, path: PathResult) -> str:
    time, walk, transfers, cost = path.costs
    return (
        f"Path {index}:\n"
        f"Time = {time:.1f} min\n"
        f"Walk = {walk:.1f} km\n"
        f"Transfers = {transfers}\n"
        f"Cost = ₹{cost:.1f}"
    )


def _add_mode_legend(ax) -> None:
    handles = [
        Line2D([0], [0], color=color, lw=3, label=mode)
        for mode, color in COLOR_MAP.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="black", lw=0, label="Source", markersize=8),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="gold",
                markeredgecolor="#7a5a00",
                color="gold",
                lw=0,
                label="Destination",
                markersize=9,
            ),
        ]
    )
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)


def _add_all_paths_legend(ax, path_count: int) -> None:
    handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle=PATH_LINE_STYLES[i % len(PATH_LINE_STYLES)], label=f"Path {i + 1}")
        for i in range(path_count)
    ]
    handles.extend(
        [
            Line2D([0], [0], color=color, lw=3, label=mode)
            for mode, color in COLOR_MAP.items()
        ]
    )
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
