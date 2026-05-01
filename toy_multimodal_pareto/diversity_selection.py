"""Top-K diverse representative selection for Pareto paths."""

from __future__ import annotations

from pareto_utils import PathResult


def objective_distance(left: PathResult, right: PathResult) -> float:
    """Manhattan distance in objective space."""

    return sum(abs(a - b) for a, b in zip(left.costs, right.costs))


class DiversePathSelector:
    """Selects anchor paths first, then greedily maximizes diversity."""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def select(self, paths: list[PathResult]) -> list[PathResult]:
        if len(paths) <= self.k:
            return list(paths)

        selected: list[PathResult] = []
        for objective_index in range(4):
            anchor = min(paths, key=lambda path: path.costs[objective_index])
            self._append_unique(selected, anchor)
            if len(selected) == self.k:
                return selected

        remaining = [path for path in paths if path not in selected]
        while remaining and len(selected) < self.k:
            candidate = max(remaining, key=lambda path: min(objective_distance(path, chosen) for chosen in selected))
            selected.append(candidate)
            remaining.remove(candidate)

        return selected

    @staticmethod
    def _append_unique(selected: list[PathResult], path: PathResult) -> None:
        if path not in selected:
            selected.append(path)
