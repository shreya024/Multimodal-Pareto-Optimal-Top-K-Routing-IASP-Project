"""Bounded multi-objective label-setting routing."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf

import networkx as nx

from pareto_utils import (
    CostVector,
    Label,
    deactivate_dominated,
    is_dominated,
    pareto_filter_paths,
    reconstruct_path,
)


@dataclass(frozen=True)
class BoundsConfig:
    """Multipliers and floors for feasibility pruning."""

    alpha_time: float = 1.8
    alpha_walk: float = 2.5
    alpha_cost: float = 3.0
    max_transfers: int = 4
    min_walk_bound: float = 8.0
    min_cost_bound: float = 220.0


@dataclass(frozen=True)
class FeasibilityBounds:
    """Absolute upper bounds used during label expansion."""

    time: float
    walk: float
    transfers: int
    cost: float
    optima: dict[str, float]


class MultiObjectiveRouter:
    """Computes Pareto-optimal paths with feasibility-bounded label setting."""

    def __init__(self, graph: nx.MultiDiGraph, bounds_config: BoundsConfig):
        self.graph = graph
        self.bounds_config = bounds_config

    def compute_bounds(self, source: int, target: int) -> FeasibilityBounds:
        optima = {
            "time": self._single_objective_shortest(source, target, "time"),
            "walk": self._single_objective_shortest(source, target, "walk"),
            "cost": self._single_objective_shortest(source, target, "cost"),
            "transfers": float(self._minimum_transfer_count(source, target)),
        }

        return FeasibilityBounds(
            time=self.bounds_config.alpha_time * optima["time"],
            walk=max(self.bounds_config.alpha_walk * optima["walk"], self.bounds_config.min_walk_bound),
            transfers=self.bounds_config.max_transfers,
            cost=max(self.bounds_config.alpha_cost * optima["cost"], self.bounds_config.min_cost_bound),
            optima=optima,
        )

    def find_pareto_paths(self, source: int, target: int) -> list:
        bounds = self.compute_bounds(source, target)
        labels_by_node: dict[int, list[Label]] = {node: [] for node in self.graph.nodes}
        start = Label(node=source, costs=(0.0, 0.0, 0, 0.0))
        labels_by_node[source].append(start)

        sequence = count()
        queue: list[tuple[float, int, Label]] = []
        heappush(queue, (0.0, next(sequence), start))

        while queue:
            _, _, label = heappop(queue)
            if not label.active:
                continue

            for _, neighbor, edge_data in self.graph.out_edges(label.node, data=True):
                if self._path_contains(label, neighbor):
                    continue

                transfer_increment = self._transfer_increment(label, edge_data)
                new_costs = self._extend_costs(label.costs, edge_data, transfer_increment)

                if not self._within_bounds(new_costs, bounds):
                    continue
                if is_dominated(new_costs, labels_by_node[neighbor]):
                    continue

                new_label = Label(
                    node=neighbor,
                    costs=new_costs,
                    prev=label,
                    previous_mode=edge_data["mode"],
                    route_id=edge_data["route_id"],
                )
                deactivate_dominated(new_costs, labels_by_node[neighbor])
                labels_by_node[neighbor].append(new_label)
                heappush(queue, (self._priority(new_costs), next(sequence), new_label))

        active_destination_labels = [label for label in labels_by_node[target] if label.active]
        return pareto_filter_paths([reconstruct_path(label) for label in active_destination_labels])

    def _single_objective_shortest(self, source: int, target: int, weight: str) -> float:
        distances = {node: inf for node in self.graph.nodes}
        distances[source] = 0.0
        queue: list[tuple[float, int]] = [(0.0, source)]

        while queue:
            distance, node = heappop(queue)
            if distance > distances[node]:
                continue
            if node == target:
                return distance

            for _, neighbor, edge_data in self.graph.out_edges(node, data=True):
                candidate = distance + float(edge_data[weight])
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    heappush(queue, (candidate, neighbor))

        raise nx.NetworkXNoPath(f"No path from {source} to {target}")

    def _minimum_transfer_count(self, source: int, target: int) -> int:
        best: dict[tuple[int, str | None, str | None], int] = {(source, None, None): 0}
        queue: list[tuple[int, int, str | None, str | None]] = [(0, source, None, None)]

        while queue:
            transfers, node, mode, route_id = heappop(queue)
            if node == target:
                return transfers
            if transfers > best[(node, mode, route_id)]:
                continue

            pseudo_label = Label(node=node, costs=(0.0, 0.0, transfers, 0.0), previous_mode=mode, route_id=route_id)
            for _, neighbor, edge_data in self.graph.out_edges(node, data=True):
                next_transfers = transfers + self._transfer_increment(pseudo_label, edge_data)
                state = (neighbor, edge_data["mode"], edge_data["route_id"])
                if next_transfers < best.get(state, inf):
                    best[state] = next_transfers
                    heappush(queue, (next_transfers, neighbor, edge_data["mode"], edge_data["route_id"]))

        raise nx.NetworkXNoPath(f"No path from {source} to {target}")

    @staticmethod
    def _extend_costs(costs: CostVector, edge_data: dict, transfer_increment: int) -> CostVector:
        return (
            costs[0] + float(edge_data["time"]),
            costs[1] + float(edge_data["walk"]),
            costs[2] + transfer_increment,
            costs[3] + float(edge_data["cost"]),
        )

    @staticmethod
    def _transfer_increment(label: Label, edge_data: dict) -> int:
        next_mode = edge_data["mode"]
        next_route = edge_data["route_id"]
        if label.previous_mode is None:
            return 0
        if label.previous_mode != next_mode:
            return 1
        if next_mode == "metro" and label.route_id != next_route:
            return 1
        return 0

    @staticmethod
    def _path_contains(label: Label, node: int) -> bool:
        cursor = label
        while cursor is not None:
            if cursor.node == node:
                return True
            cursor = cursor.prev
        return False

    @staticmethod
    def _within_bounds(costs: CostVector, bounds: FeasibilityBounds) -> bool:
        return (
            costs[0] <= bounds.time
            and costs[1] <= bounds.walk
            and costs[2] <= bounds.transfers
            and costs[3] <= bounds.cost
        )

    @staticmethod
    def _priority(costs: CostVector) -> float:
        return costs[0] + costs[1] + costs[2] + costs[3]
