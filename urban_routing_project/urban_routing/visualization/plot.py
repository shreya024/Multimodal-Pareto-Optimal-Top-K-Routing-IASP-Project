"""
visualization/plot.py — Pareto front plots and route comparison charts.

Generates:
  1. 2-D scatter: time vs cost Pareto front with selected Top-K highlighted
  2. Parallel coordinates: all objectives across selected routes
  3. Bar chart: method comparison (runtime, diversity, hypervolume)
  4. Route map: geographic plot of selected routes on Bengaluru grid
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from data.schema import ParetoPath
import config as cfg

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Colour helpers ─────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "Pareto+Diversity": "#2196F3",
    "Pareto+Cluster":   "#4CAF50",
    "WeightedSum":      "#FF9800",
    "Lexicographic":    "#E91E63",
}


# ── 1. Pareto scatter (time vs cost) ─────────────────────────────────────────

def plot_pareto_front(
    pareto_paths:  List[ParetoPath],
    selected:      List[ParetoPath] = None,
    title:         str = "Pareto Front: Time vs Cost",
    save_path:     Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    times = [p.total_weight.time_min  for p in pareto_paths]
    costs = [p.total_weight.cost_inr  for p in pareto_paths]

    ax.scatter(times, costs, c="#B0BEC5", s=30, alpha=0.6, label=f"Pareto front (n={len(pareto_paths)})")

    if selected:
        st = [p.total_weight.time_min for p in selected]
        sc = [p.total_weight.cost_inr for p in selected]
        ax.scatter(st, sc, c="#F44336", s=120, zorder=5, marker="*",
                   edgecolors="black", linewidths=0.5, label=f"Selected Top-{len(selected)}")
        for i, (t, c) in enumerate(zip(st, sc)):
            ax.annotate(f"R{i+1}", (t, c), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Travel Time (min)", fontsize=11)
    ax.set_ylabel("Cost (₹)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, save_path or OUTPUT_DIR / "pareto_front.png")


# ── 2. Parallel coordinates ───────────────────────────────────────────────────

def plot_parallel_coordinates(
    paths:    List[ParetoPath],
    labels:   List[str] = None,
    title:    str = "Selected Routes — Parallel Coordinates",
    save_path: Optional[Path] = None,
):
    if not paths:
        return

    vecs = np.array([p.total_weight.as_tuple() for p in paths])
    n, d = vecs.shape

    # Normalize each column
    mins = vecs.min(axis=0);  maxs = vecs.max(axis=0)
    rngs = maxs - mins;       rngs[rngs == 0] = 1.0
    norm = (vecs - mins) / rngs

    colors = cm.tab10(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(10, 5))

    x = list(range(d))
    for i, (row, color) in enumerate(zip(norm, colors)):
        lbl = labels[i] if labels else f"Route {i+1}"
        ax.plot(x, row, "o-", color=color, linewidth=2, markersize=7, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(cfg.OBJECTIVE_NAMES, fontsize=10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Best", "", "Mid", "", "Worst"])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path or OUTPUT_DIR / "parallel_coords.png")


# ── 3. Method comparison bar chart ───────────────────────────────────────────

def plot_method_comparison(
    results:   List[Dict],
    save_path: Optional[Path] = None,
):
    """
    results: list of dicts from benchmark.run_benchmark()
    """
    methods     = list(dict.fromkeys(r["method"] for r in results))
    metrics_map = {k: [] for k in ["runtime_s", "diversity_score", "hypervolume"]}

    for method in methods:
        rows = [r for r in results if r["method"] == method]
        for metric in metrics_map:
            vals = [r.get(metric, 0) for r in rows if r.get(metric) is not None]
            metrics_map[metric].append(np.mean(vals) if vals else 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metric_labels = {
        "runtime_s":      "Runtime (s)\n[lower = better]",
        "diversity_score": "Structural Diversity\n[higher = better]",
        "hypervolume":    "Hypervolume\n[higher = better]",
    }

    for ax, (metric, label) in zip(axes, metric_labels.items()):
        vals   = metrics_map[metric]
        colors = [METHOD_COLORS.get(m, "#9E9E9E") for m in methods]
        bars   = ax.bar(methods, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Method Comparison Across All OD Pairs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save_path or OUTPUT_DIR / "method_comparison.png")


# ── 4. Geographic route overlay ───────────────────────────────────────────────

def plot_routes_on_map(
    paths:     List[ParetoPath],
    stops_dict: Dict,
    labels:    List[str] = None,
    title:     str = "Selected Routes (Geographic)",
    save_path:  Optional[Path] = None,
):
    from data.schema import TransportMode
    fig, ax = plt.subplots(figsize=(9, 9))

    # Background: all stops as grey dots
    lats = [s.lat for s in stops_dict.values()]
    lons = [s.lon for s in stops_dict.values()]
    ax.scatter(lons, lats, c="#CFD8DC", s=8, zorder=1)

    colors = cm.tab10(np.linspace(0, 1, len(paths)))

    for i, (path, color) in enumerate(zip(paths, colors)):
        lbl = labels[i] if labels else f"Route {i+1}"
        edge_lons, edge_lats = [], []
        for edge in path.edges:
            src = stops_dict.get(edge.src)
            dst = stops_dict.get(edge.dst)
            if src and dst:
                edge_lons += [src.lon, dst.lon, None]
                edge_lats += [src.lat, dst.lat, None]
        # Strip trailing None
        while edge_lons and edge_lons[-1] is None:
            edge_lons.pop(); edge_lats.pop()

        ax.plot(edge_lons, edge_lats, "-", color=color, linewidth=2.5,
                alpha=0.8, label=lbl, zorder=2)

        # Mark origin and destination
        if path.nodes:
            for nid, marker, ms in [(path.nodes[0], "o", 10), (path.nodes[-1], "s", 10)]:
                s = stops_dict.get(nid)
                if s:
                    ax.scatter(s.lon, s.lat, c=[color], s=ms**2, zorder=3,
                               edgecolors="black", linewidths=0.8)

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save_or_show(fig, save_path or OUTPUT_DIR / "route_map.png")


# ── Objective radar chart ─────────────────────────────────────────────────────

def plot_radar(
    paths:    List[ParetoPath],
    labels:   List[str] = None,
    title:    str = "Route Objective Radar",
    save_path: Optional[Path] = None,
):
    if not paths:
        return

    vecs = np.array([p.total_weight.as_tuple() for p in paths])
    mins = vecs.min(axis=0);  maxs = vecs.max(axis=0)
    rngs = maxs - mins;       rngs[rngs == 0] = 1.0
    # Invert so "better" = larger radius
    norm = 1.0 - (vecs - mins) / rngs

    n_obj    = vecs.shape[1]
    angles   = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors   = cm.tab10(np.linspace(0, 1, len(paths)))

    for i, (row, color) in enumerate(zip(norm, colors)):
        vals = row.tolist() + row[:1].tolist()
        lbl  = labels[i] if labels else f"R{i+1}"
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=lbl)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cfg.OBJECTIVE_NAMES, size=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Worst 25%", "Mid", "Good", "Best"], size=7)
    ax.set_title(title, size=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path or OUTPUT_DIR / "radar.png")


# ── Internal ──────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}")
