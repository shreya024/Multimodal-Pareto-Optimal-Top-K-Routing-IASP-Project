"""
core/label.py — Pareto label: a cost vector accumulated along a path.

A label L = (cost_vector, path_info) at node v represents the best known
multi-dimensional cost of reaching v via a specific partial path.

Dominance: label A dominates B iff
  A[i] <= B[i] for all i  AND  A[i] < B[i] for some i
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config as cfg
from data.schema import EdgeWeight, GraphEdge


@dataclass
class Label:
    """
    A non-dominated label at a graph node.

    cost    — accumulated EdgeWeight (multi-dimensional)
    node    — current node (stop_id)
    prev    — predecessor Label (for path reconstruction)
    edge    — edge taken to reach this node from prev
    """
    cost:   EdgeWeight
    node:   str
    prev:   Optional["Label"] = field(default=None, repr=False)
    edge:   Optional[GraphEdge] = field(default=None, repr=False)

    # ── Dominance ─────────────────────────────────────────────────────────────

    def dominates(self, other: "Label") -> bool:
        """Return True if self Pareto-dominates other."""
        a = self.cost.as_tuple()
        b = other.cost.as_tuple()
        at_least_as_good = all(x <= y for x, y in zip(a, b))
        strictly_better  = any(x <  y for x, y in zip(a, b))
        return at_least_as_good and strictly_better

    def is_dominated_by(self, other: "Label") -> bool:
        return other.dominates(self)

    # ── Priority queue ordering (min-heap on primary objective) ───────────────

    def priority(self) -> float:
        return self.cost.as_tuple()[cfg.PARETO_HEAP_TIE_BREAKER]

    def __lt__(self, other: "Label") -> bool:
        return self.priority() < other.priority()

    def __le__(self, other: "Label") -> bool:
        return self.priority() <= other.priority()

    # ── Path reconstruction ───────────────────────────────────────────────────

    def reconstruct_path(self) -> Tuple[List[str], List[GraphEdge]]:
        """Walk predecessor chain to rebuild the full node/edge sequence."""
        nodes: List[str] = []
        edges: List[GraphEdge] = []
        cur = self
        while cur is not None:
            nodes.append(cur.node)
            if cur.edge is not None:
                edges.append(cur.edge)
            cur = cur.prev
        nodes.reverse()
        edges.reverse()
        return nodes, edges

    def __repr__(self) -> str:
        c = self.cost
        return (
            f"Label(node={self.node}, "
            f"t={c.time_min:.1f}m, ₹{c.cost_inr:.1f}, "
            f"tr={c.transfers:.0f}, walk={c.walking_m:.0f}m, co2={c.co2_g:.0f}g)"
        )


# ── Frontier (set of non-dominated labels at one node) ───────────────────────

class LabelFrontier:
    """
    Maintains the Pareto frontier of labels at a single node.
    Only non-dominated labels are retained.
    Capped at MAX_LABELS_PER_NODE to prevent combinatorial explosion.
    """

    def __init__(self):
        self._labels: List[Label] = []

    def try_add(self, new_label: Label) -> bool:
        """
        Attempt to add new_label to the frontier.
        Returns True if the label was added (i.e. it was non-dominated
        and may propagate further).
        Removes any previously stored labels that are now dominated by new_label.
        """
        # Check if new_label is dominated by any existing label
        for existing in self._labels:
            if existing.dominates(new_label):
                return False  # new_label is dominated; discard

        # Remove labels dominated by new_label
        self._labels = [lb for lb in self._labels if not new_label.dominates(lb)]

        # Hard cap
        if len(self._labels) >= cfg.MAX_LABELS_PER_NODE:
            return False

        self._labels.append(new_label)
        return True

    def labels(self) -> List[Label]:
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._labels)
