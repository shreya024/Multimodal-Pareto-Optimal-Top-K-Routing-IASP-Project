"""
tests/test_dominance.py — Unit tests for Pareto dominance logic.
"""
import pytest
from algorithms.dominance import (
    dominates, is_non_dominated, prune_to_pareto_front, fast_pareto_filter
)
import numpy as np


class TestDominates:

    def test_clear_dominance(self):
        # a is strictly better on all dims
        assert dominates((1, 2, 3), (2, 3, 4))

    def test_not_dominates_equal(self):
        # Equal vectors: neither dominates the other
        assert not dominates((1, 2, 3), (1, 2, 3))

    def test_not_dominates_mixed(self):
        # a better on dim 0 but worse on dim 1
        assert not dominates((1, 5, 3), (2, 3, 4))

    def test_dominates_partial_equal(self):
        # Better on some, equal on others — still dominates
        assert dominates((1, 2, 3), (2, 2, 4))

    def test_dominates_single_dim(self):
        assert dominates((1,), (2,))
        assert not dominates((2,), (1,))


class TestNonDominated:

    def test_non_dominated_in_empty_frontier(self):
        assert is_non_dominated((1, 2, 3), [])

    def test_dominated(self):
        frontier = [(0, 1, 2), (1, 0, 2)]
        assert not is_non_dominated((2, 3, 4), frontier)

    def test_truly_non_dominated(self):
        frontier = [(2, 1, 3), (3, 2, 1)]
        assert is_non_dominated((1, 2, 2), frontier)  # trade-off on dim 0 vs 1


class TestPruneFront:

    def test_trivial_front(self):
        vecs = [(1, 10), (5, 2), (3, 5)]
        idx  = prune_to_pareto_front(vecs)
        # All three are non-dominated
        assert set(idx) == {0, 1, 2}

    def test_one_dominated(self):
        vecs = [(1, 10), (5, 2), (3, 5), (6, 8)]
        idx  = prune_to_pareto_front(vecs)
        assert 3 not in idx    # (6,8) is dominated by (5,2) and (3,5)

    def test_single_vector(self):
        assert prune_to_pareto_front([(3, 4)]) == [0]


class TestFastParetoFilter:

    def test_matches_naive(self):
        np.random.seed(7)
        mat = np.random.rand(50, 4)
        fast_mask  = fast_pareto_filter(mat)
        naive_idx  = set(prune_to_pareto_front([tuple(r) for r in mat]))
        fast_idx   = set(np.where(fast_mask)[0])
        assert fast_idx == naive_idx
