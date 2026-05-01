"""Synthetic multimodal transportation graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from random import Random

import networkx as nx

from cost_models import (
    MetroCostConfig,
    RoadCostConfig,
    WalkingCostConfig,
    metro_cost,
    metro_time,
    road_cost,
    road_time,
    walking_cost,
    walking_time,
)


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for synthetic city generation."""

    grid_size: int = 6
    spacing_km: float = 1.0
    walking_radius_km: float = 1.05
    congestion_min: float = 1.0
    congestion_max: float = 1.5
    random_seed: int = 7
    metro_rows: tuple[int, ...] = (1, 4)
    metro_cols: tuple[int, ...] = (2,)
    road: RoadCostConfig = field(default_factory=RoadCostConfig)
    metro: MetroCostConfig = field(default_factory=MetroCostConfig)
    walking: WalkingCostConfig = field(default_factory=WalkingCostConfig)


class SyntheticGraphBuilder:
    """Builds a deterministic, layered multimodal synthetic city graph."""

    def __init__(self, config: GraphConfig):
        if config.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        self.config = config
        self._random = Random(config.random_seed)

    def build(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        self._add_nodes(graph)
        self._add_road_edges(graph)
        self._add_walking_edges(graph)
        self._add_metro_edges(graph)
        return graph

    def _node_id(self, row: int, col: int) -> int:
        return row * self.config.grid_size + col

    def _add_nodes(self, graph: nx.MultiDiGraph) -> None:
        for row in range(self.config.grid_size):
            for col in range(self.config.grid_size):
                node = self._node_id(row, col)
                graph.add_node(node, pos=(col * self.config.spacing_km, row * self.config.spacing_km))

    def _distance(self, node_a: int, node_b: int) -> float:
        ax, ay = self._position(node_a)
        bx, by = self._position(node_b)
        return hypot(ax - bx, ay - by)

    def _position(self, node: int) -> tuple[float, float]:
        return self._node_pos(node)

    def _node_pos(self, node: int) -> tuple[float, float]:
        row, col = divmod(node, self.config.grid_size)
        return col * self.config.spacing_km, row * self.config.spacing_km

    def _add_bidirectional_edge(
        self,
        graph: nx.MultiDiGraph,
        u: int,
        v: int,
        *,
        time: float,
        walk: float,
        mode: str,
        route_id: str,
        cost: float,
    ) -> None:
        attrs = {
            "time": float(time),
            "walk": float(walk),
            "mode": mode,
            "route_id": route_id,
            "cost": float(cost),
        }
        graph.add_edge(u, v, **attrs)
        graph.add_edge(v, u, **attrs)

    def _add_road_edges(self, graph: nx.MultiDiGraph) -> None:
        for row in range(self.config.grid_size):
            for col in range(self.config.grid_size):
                current = self._node_id(row, col)
                for next_row, next_col in ((row + 1, col), (row, col + 1)):
                    if next_row >= self.config.grid_size or next_col >= self.config.grid_size:
                        continue
                    neighbor = self._node_id(next_row, next_col)
                    distance = self._distance(current, neighbor)
                    congestion = self._random.uniform(self.config.congestion_min, self.config.congestion_max)
                    time = road_time(distance, congestion, self.config.road)
                    cost = road_cost(distance, time, congestion, self.config.road)
                    self._add_bidirectional_edge(
                        graph,
                        current,
                        neighbor,
                        time=time,
                        walk=0.0,
                        mode="road",
                        route_id="road_grid",
                        cost=cost,
                    )

    def _add_walking_edges(self, graph: nx.MultiDiGraph) -> None:
        nodes = list(graph.nodes)
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                distance = self._distance(u, v)
                if distance <= self.config.walking_radius_km:
                    self._add_bidirectional_edge(
                        graph,
                        u,
                        v,
                        time=walking_time(distance, self.config.walking),
                        walk=distance,
                        mode="walk",
                        route_id="walk",
                        cost=walking_cost(),
                    )

    def _add_metro_edges(self, graph: nx.MultiDiGraph) -> None:
        for row in self.config.metro_rows:
            if 0 <= row < self.config.grid_size:
                route_id = f"metro_row_{row}"
                nodes = [self._node_id(row, col) for col in range(self.config.grid_size)]
                self._add_metro_line(graph, nodes, route_id)

        for col in self.config.metro_cols:
            if 0 <= col < self.config.grid_size:
                route_id = f"metro_col_{col}"
                nodes = [self._node_id(row, col) for row in range(self.config.grid_size)]
                self._add_metro_line(graph, nodes, route_id)

    def _add_metro_line(self, graph: nx.MultiDiGraph, nodes: list[int], route_id: str) -> None:
        for u, v in zip(nodes, nodes[1:]):
            distance = self._distance(u, v)
            self._add_bidirectional_edge(
                graph,
                u,
                v,
                time=metro_time(distance, self.config.metro),
                walk=0.0,
                mode="metro",
                route_id=route_id,
                cost=metro_cost(distance, self.config.metro),
            )
