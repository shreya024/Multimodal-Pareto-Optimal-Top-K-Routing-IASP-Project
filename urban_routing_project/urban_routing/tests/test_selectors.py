"""
tests/test_selectors.py — Unit tests for Top-K selection strategies.
"""
import pytest
import numpy as np
from data.schema import EdgeWeight, GraphEdge, ParetoPath, TransportMode
from selection.diversity_selector import DiversitySelector, jaccard_dissimilarity
from selection.cluster_selector import ClusterSelector


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_path(time, cost, transfers, walking, co2, edges=None):
    """Construct a synthetic ParetoPath with given objective values."""
    w = EdgeWeight(
        time_min=time, cost_inr=cost, transfers=transfers,
        walking_m=walking, co2_g=co2
    )
    return ParetoPath(nodes=[], edges=edges or [], total_weight=w)


def make_edge(src, dst, route_id):
    return GraphEdge(
        src=src, dst=dst, route_id=route_id,
        mode=TransportMode.BUS,
        weight=EdgeWeight.zero(),
        distance_m=0.0,
    )


# ── Jaccard tests ──────────────────────────────────────────────────────────────

class TestJaccardDissimilarity:

    def test_identical_paths(self):
        edges = [make_edge("A", "B", "R1"), make_edge("B", "C", "R1")]
        p1 = make_path(10, 5, 0, 100, 500, edges)
        p2 = make_path(12, 6, 0, 120, 550, list(edges))
        assert jaccard_dissimilarity(p1, p2) == pytest.approx(0.0)

    def test_disjoint_paths(self):
        e1 = [make_edge("A", "B", "R1")]
        e2 = [make_edge("C", "D", "R2")]
        p1 = make_path(10, 5, 0, 100, 500, e1)
        p2 = make_path(10, 5, 0, 100, 500, e2)
        assert jaccard_dissimilarity(p1, p2) == pytest.approx(1.0)

    def test_partial_overlap(self):
        shared = make_edge("A", "B", "R1")
        e1 = [shared, make_edge("B", "C", "R1")]
        e2 = [shared, make_edge("B", "D", "R2")]
        p1 = make_path(10, 5, 0, 0, 0, e1)
        p2 = make_path(15, 7, 0, 0, 0, e2)
        d  = jaccard_dissimilarity(p1, p2)
        # |intersection|=1, |union|=3  → d = 1 - 1/3 = 2/3
        assert d == pytest.approx(2 / 3, abs=1e-6)


# ── DiversitySelector tests ────────────────────────────────────────────────────

class TestDiversitySelector:

    def _make_pareto_set(self, n=20):
        """Generate a synthetic Pareto set along a time-cost tradeoff."""
        paths = []
        for i in range(n):
            t    = 10 + i * 5       # 10..105 min
            c    = 100 - i * 4      # 100..20 INR
            tr   = i % 3
            walk = 100 + i * 20
            co2  = 200 + i * 10
            # Give each path unique edges so Jaccard doesn't collapse
            edges = [make_edge(f"S{i}", f"S{i+1}", f"R{i}")]
            paths.append(make_path(t, c, tr, walk, co2, edges))
        return paths

    def test_returns_k_paths(self):
        paths = self._make_pareto_set(20)
        sel   = DiversitySelector(k=5)
        result = sel.select(paths)
        assert len(result) == 5

    def test_returns_all_if_fewer_than_k(self):
        paths  = self._make_pareto_set(3)
        sel    = DiversitySelector(k=5)
        result = sel.select(paths)
        assert len(result) == 3

    def test_empty_input(self):
        sel = DiversitySelector(k=5)
        assert sel.select([]) == []

    def test_selected_are_subset(self):
        paths  = self._make_pareto_set(20)
        sel    = DiversitySelector(k=5)
        result = sel.select(paths)
        for r in result:
            assert r in paths

    def test_higher_diversity_than_naive_first_k(self):
        """Diversity selector should beat naively taking first K paths."""
        from evaluation.metrics import diversity_score as div_score
        paths   = self._make_pareto_set(20)
        sel     = DiversitySelector(k=5)
        smart   = sel.select(paths)
        naive   = paths[:5]
        assert div_score(smart) >= div_score(naive) - 0.01  # allow tiny slack


# ── ClusterSelector tests ──────────────────────────────────────────────────────

class TestClusterSelector:

    def _make_pareto_set(self, n=20):
        return [
            make_path(10 + i * 4, 80 - i * 3, i % 4, 50 + i * 30, 300 + i * 15)
            for i in range(n)
        ]

    def test_returns_k_paths(self):
        paths  = self._make_pareto_set(20)
        sel    = ClusterSelector(k=5)
        result = sel.select(paths)
        assert len(result) == 5

    def test_selected_are_subset(self):
        paths  = self._make_pareto_set(20)
        sel    = ClusterSelector(k=5)
        result = sel.select(paths)
        for r in result:
            assert r in paths

    def test_empty(self):
        assert ClusterSelector(k=3).select([]) == []

    def test_k_larger_than_set(self):
        paths  = self._make_pareto_set(3)
        result = ClusterSelector(k=10).select(paths)
        assert len(result) == 3
