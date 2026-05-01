# 5x5 Multimodal Pareto Routing Demo

This miniature network mirrors the presentation narrative: a grid road network, straight metro lines, and walking/access links between nearby nodes.

- Origin: `R_0_0`
- Destination: `R_4_4`
- Nodes: `35`
- Edges: `198`
- Pareto frontier size: `17`

The demo is deliberately small so the report can explain why scalar shortest path is insufficient: the fastest, cheapest, lowest-walk, and lowest-transfer paths are not the same route.

## Diversity-Constrained Top-K

| Route | Time | Walk | Transfers | Cost | CO2 | Hops | Modes |
|---|---:|---:|---:|---:|---:|---:|---|
| R1 | 42.0 | 240 | 2 | 48.0 | 2200 | 10 | bus + transfer + metro |
| R2 | 96.0 | 4000 | 0 | 0.0 | 0 | 8 | walk |
| R3 | 66.0 | 1500 | 0 | 40.0 | 2250 | 8 | bus + walk |
| R4 | 75.0 | 2740 | 2 | 12.0 | 300 | 10 | walk + transfer + metro |
| R5 | 48.0 | 0 | 0 | 64.0 | 3600 | 8 | bus |

## Cluster Top-K

| Route | Time | Walk | Transfers | Cost | CO2 | Hops | Modes |
|---|---:|---:|---:|---:|---:|---:|---|
| R1 | 48.0 | 740 | 2 | 40.0 | 1750 | 10 | bus + transfer + metro + walk |
| R2 | 90.0 | 3500 | 0 | 8.0 | 450 | 8 | bus + walk |
| R3 | 54.0 | 500 | 0 | 56.0 | 3150 | 8 | bus + walk |
| R4 | 75.0 | 2740 | 2 | 12.0 | 300 | 10 | walk + transfer + metro |
| R5 | 72.0 | 2000 | 0 | 32.0 | 1800 | 8 | bus + walk |

## Weighted-Sum Baseline

| Route | Time | Walk | Transfers | Cost | CO2 | Hops | Modes |
|---|---:|---:|---:|---:|---:|---:|---|
| R1 | 48.0 | 0 | 0 | 64.0 | 3600 | 8 | bus |

## Lexicographic Baseline

| Route | Time | Walk | Transfers | Cost | CO2 | Hops | Modes |
|---|---:|---:|---:|---:|---:|---:|---|
| R1 | 42.0 | 240 | 2 | 48.0 | 2200 | 10 | bus + transfer + metro |

## Metrics: Diversity Top-K

- Paths: `5`
- Diversity score: `0.856`
- Hypervolume: `1.77e+10`
- Time spread: `0.686`

## Metrics: Cluster Top-K

- Paths: `5`
- Diversity score: `0.815`
- Hypervolume: `7.61e+09`
- Time spread: `0.629`

## Metrics: Weighted Sum

- Paths: `1`
- Diversity score: `0.000`
- Hypervolume: `0.00e+00`
- Time spread: `0.000`

## Metrics: Lexicographic

- Paths: `1`
- Diversity score: `0.000`
- Hypervolume: `0.00e+00`
- Time spread: `0.000`
