# Urban Multimodal Pareto-Optimal Routing

A full implementation of multi-criteria Pareto-optimal route planning on the BMTC (Bengaluru Metropolitan Transport Corporation) bus dataset.

## Architecture

```
urban_routing/
├── data/
│   ├── loader.py          # BMTC dataset ingestion & cleaning
│   └── schema.py          # Typed data structures
├── core/
│   ├── graph.py           # Multi-layer multigraph construction
│   ├── edge_weights.py    # Multi-dimensional weight vector computation
│   └── label.py           # Pareto label (dominance, merging)
├── algorithms/
│   ├── pareto_dijkstra.py # Multi-criteria Dijkstra for Pareto frontier
│   └── dominance.py       # Dominance checks & frontier maintenance
├── selection/
│   ├── diversity_selector.py  # Jaccard-diversity Top-K selector
│   └── cluster_selector.py    # Clustering-based centroid selector
├── baselines/
│   ├── weighted_sum.py    # Weighted-sum scalarization baseline
│   └── lexicographic.py   # Lexicographic ordering baseline
├── evaluation/
│   ├── metrics.py         # Hypervolume, spread, diversity metrics
│   └── benchmark.py       # Runtime & quality comparison harness
├── visualization/
│   └── plot.py            # Pareto front & route plots
├── tests/
│   ├── test_dominance.py
│   ├── test_pareto_dijkstra.py
│   └── test_selectors.py
├── config.py              # Global configuration
├── main.py                # CLI entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download the BMTC dataset from Kaggle:
https://www.kaggle.com/datasets/shivamishra2112/bmtc-bus-transportation-dataset

Place the CSV files in `data/raw/`.

## Usage

```bash
# Run full pipeline: build graph, compute Pareto frontier, select Top-K, compare baselines
python main.py --origin "Majestic" --destination "Whitefield" --top-k 5

# Run benchmarks
python main.py --benchmark

# Use synthetic data (no download needed)
python main.py --synthetic --origin 0 --destination 99 --top-k 5
```

## Objectives

| Objective     | Unit        | Description                          |
|--------------|-------------|--------------------------------------|
| Time         | minutes     | Total journey time incl. transfers   |
| Cost         | INR         | Fare across all segments             |
| Transfers    | count       | Number of modal/route switches       |
| Walking      | meters      | Total walking distance               |
| CO2          | grams       | Estimated emissions                  |

## Methodology

1. **Multi-layer graph**: Road + metro + walking layers fused into a directed multigraph  
2. **Multi-criteria Dijkstra**: Non-dominated label sets propagated per node  
3. **Top-K selection**: Diversity-constrained (Jaccard) vs clustering (k-means on objective space)  
4. **Baselines**: Weighted-sum scalarization & lexicographic ordering  
