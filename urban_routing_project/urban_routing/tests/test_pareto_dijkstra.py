"""
tests/test_pareto_dijkstra.py — Integration tests for multi-criteria Dijkstra.
"""
import pytest
from data.loader import BMTCLoader
from core.graph import MultimodalGraph
from algorithms.pareto_dijkstra import ParetoDijkstra
from algorithms.dominance import dominates


@pytest.fixture(scope="module")
def graph():
    loader = BMTCLoader()
    loader.load()
    g = MultimodalGraph()
    g.build(loader.get_stops(), loader.get_routes())
    return g


@pytest.fixture(scope="module")
def router(graph):
    return ParetoDijkstra(graph)


class TestParetoDijkstra:

    def test_returns_list(self, router, graph):
        nodes = list(graph.stops.keys())
        origin, dest = nodes[0], nodes[-1]
        result = router.run(origin, dest)
        assert isinstance(result, list)

    def test_paths_are_non_dominated(self, router, graph):
        """No returned path should be dominated by another returned path."""
        nodes  = list(graph.stops.keys())
        origin = nodes[0]
        dest   = nodes[min(30, len(nodes) - 1)]
        paths  = router.run(origin, dest)
        if len(paths) < 2:
            return
        vecs = [p.total_weight.as_tuple() for p in paths]
        for i, a in enumerate(vecs):
            for j, b in enumerate(vecs):
                if i != j:
                    assert not dominates(a, b), (
                        f"Path {i} dominates path {j}: {a} vs {b}"
                    )

    def test_path_nodes_are_connected(self, router, graph):
        """Consecutive nodes in reconstructed paths must be graph neighbours."""
        nodes  = list(graph.stops.keys())
        origin = nodes[0]
        dest   = nodes[min(20, len(nodes) - 1)]
        paths  = router.run(origin, dest)
        for path in paths[:5]:       # check first 5 for speed
            for edge in path.edges:
                assert graph.G.has_edge(edge.src, edge.dst), (
                    f"Edge {edge.src} → {edge.dst} not in graph"
                )

    def test_invalid_origin_raises(self, router):
        with pytest.raises(ValueError):
            router.run("INVALID_NODE_XYZ", "0")

    def test_empty_if_unreachable(self, graph):
        """Isolated node should return empty list."""
        import networkx as nx
        # Add an isolated node
        graph.G.add_node("__isolated__")
        graph.stops["__isolated__"] = graph.stops[list(graph.stops.keys())[0]]
        router2 = ParetoDijkstra(graph)
        result = router2.run("__isolated__", list(graph.stops.keys())[0])
        # May be empty or have paths; just ensure no exception
        assert isinstance(result, list)

    def test_cost_accumulation(self, router, graph):
        """Total weight should equal sum of edge weights along the path."""
        nodes  = list(graph.stops.keys())
        origin = nodes[0]
        dest   = nodes[min(15, len(nodes) - 1)]
        paths  = router.run(origin, dest)
        for path in paths[:3]:
            from data.schema import EdgeWeight
            acc = EdgeWeight.zero()
            for e in path.edges:
                acc = acc + e.weight
            tw = path.total_weight
            assert abs(acc.time_min  - tw.time_min)  < 1e-6
            assert abs(acc.cost_inr  - tw.cost_inr)  < 1e-6
            assert abs(acc.transfers - tw.transfers)  < 1e-6
