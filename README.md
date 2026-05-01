# Multimodal Pareto-Optimal Top-K Routing

**Course Project for SL 203: Introduction to Algorithms and Software Programming**  
**Instructor:** Dr. Nagarajan Natrajan  

**Team:**
- Shreya Ghosh (25822)
- Soham Chakraborty (25845)
- Rahul Dev Sharma (26845)

## 📖 Overview

This repository contains the implementation, experiments, and comprehensive research report for the **Multimodal Pareto-Optimal Top-K Routing** system. Standard shortest-path algorithms (like Dijkstra's) typically collapse multiple routing metrics into a single weighted score, obscuring valuable alternatives. This project implements a **Multi-Objective Label-Setting Algorithm (Pareto Dijkstra)** that evaluates multiple conflicting objectives simultaneously:

| Objective     | Unit        | Description                          |
|--------------|-------------|--------------------------------------|
| Time         | minutes     | Total journey time incl. transfers   |
| Cost         | INR         | Fare across all segments             |
| Transfers    | count       | Number of modal/route switches       |
| Walking      | meters      | Total walking distance               |
| CO2          | grams       | Estimated emissions                  |

By computing the Pareto frontier, the system offers users a set of mathematically non-dominated paths. A greedy **Top-K Diversity Selector** based on Jaccard dissimilarity is then applied to extract the most structurally diverse routes (e.g., "Fastest", "Cheapest", "Eco-Friendly").

## 📂 Repository Structure & Architecture

The project is split into synthetic testing environments, complex real-world routing, and comprehensive documentation:

- **`toy_multimodal_pareto/`**: A synthetic, controlled grid-network environment featuring road networks, straight metro lines, and walking links. This serves as an intuitive testbed to validate label expansion, dominance pruning, and visual outputs.
- **`urban_routing_project/`**: A full implementation of multi-criteria Pareto-optimal route planning on the BMTC (Bengaluru Metropolitan Transport Corporation) bus dataset.
  ```text
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
- **`report/`**: A comprehensive research report documenting the system architecture, mathematical problem formulation, algorithmic design, complexity analysis, and experimental results.

## 🚀 Key Features

- **Multi-layer Graph Construction**: Road + metro + walking layers fused into a unified directed multigraph.
- **Multi-criteria Dijkstra**: Non-dominated label sets propagated per node efficiently to prevent combinatorial explosion.
- **Label Frontier Management**: Utilizes an $F_{max}$ cap per node to bound memory complexity in dense urban networks.
- **Top-K Selection**: Employs diversity-constrained selection (Jaccard) and clustering-based selection (k-means on objective space) vs baselines (Weighted-sum scalarization & lexicographic ordering).
- **Rich Visualizations**: Generates intuitive visual outputs including interactive route maps, time-cost Pareto fronts, parallel coordinate plots, and radar (spider) charts.
- **External Contribution**: As part of this coursework, we successfully contributed a **[Pointer Visualization](https://github.com/Kuldeep1709/C-Code-Visualizer/pull/1)** feature to the open-source [C-Code-Visualizer](https://github.com/Kuldeep1709/C-Code-Visualizer) project to aid in DSA and systems debugging.

## 🏃 Dataset & Setup

### Requirements
Make sure you have Python installed. You can install the required dependencies by navigating to the respective project directory and running:
```bash
pip install -r requirements.txt
```

### Dataset Download
For the real-world urban network, download the **BMTC dataset** from Kaggle:
[BMTC Bus Transportation Dataset](https://www.kaggle.com/datasets/shivamishra2112/bmtc-bus-transportation-dataset)
Place the extracted CSV files into `urban_routing_project/urban_routing/data/raw/`.

## 💻 Usage

### 1. Toy Multimodal Network
To run the synthetic testbed and generate the visual outputs (Pareto front, path graphs):
```bash
cd toy_multimodal_pareto
python main.py
```

### 2. Urban Routing Network
To run the complex real-world urban routing simulation:
```bash
cd urban_routing_project/urban_routing

# Run full pipeline: build graph, compute Pareto frontier, select Top-K, compare baselines
python main.py --origin "Majestic" --destination "Whitefield" --top-k 5

# Run benchmarks
python main.py --benchmark

# Use synthetic data (no download needed)
python main.py --synthetic --origin 0 --destination 99 --top-k 5
```

## 📄 Documentation

The full research report (`report.pdf`) is available in the `report/` directory. It serves as the definitive documentation for the mathematics, system architecture, and experimental results of this project.

## 🛠 Tech Stack

- **Algorithms:** Multi-objective Dijkstra, Max-Min Greedy Diversity Selection, k-means Clustering
- **Data Processing:** Python, NetworkX, Pandas, NumPy
