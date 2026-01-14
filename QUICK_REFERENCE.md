# Quick Reference Guide

## Project Structure

```
chess-network-analysis/
├── README.md                    # Main project documentation
├── METHODOLOGY.md               # Technical details and algorithms
├── CODE_MAPPING.md              # Maps old code to new structure
├── QUICK_REFERENCE.md           # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── data/
│   ├── raw/                     # Original data files (not in git)
│   └── processed/               # Cleaned data (not in git)
│
├── src/                         # Main source code
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning and standardization
│   ├── network_construction.py  # Network building and topology
│   ├── ego_network_analysis.py  # Player-specific networks
│   └── geographic_analysis.py   # Geographic dispersion
│
├── examples/
│   └── quick_start.py           # Basic usage example
│
├── results/
│   ├── figures/                 # Generated plots
│   └── tables/                  # Statistical results
│
└── redes/                       # Original messy code (for reference)
```

## Common Tasks

### 1. Load and Clean Data

```python
from src.data_preprocessing import clean_chess_data, filter_by_years

# Clean raw data
df = clean_chess_data('data/raw/chess_games.csv')

# Filter by date range
df_2000s = filter_by_years(df, 2000, 2020)
```

### 2. Build Network

```python
from src.data_preprocessing import create_player_table
from src.network_construction import build_chess_network

# Create player table
players = create_player_table(df_2000s)

# Build network
G = build_chess_network(df_2000s, players)
```

### 3. Analyze Network Topology

```python
from src.network_construction import analyze_topology, calculate_assortativity

# Get basic metrics
metrics = analyze_topology(G)
print(f"Nodes: {metrics['nodes']}, Edges: {metrics['edges']}")
print(f"Density: {metrics['density']:.6f}")

# Calculate assortativity
assortativity = calculate_assortativity(G)
print(f"Assortativity coefficient: {assortativity['mu']:.4f}")
```

### 4. Identify Hubs

```python
from src.network_construction import identify_hubs

# Find hub players (top 5%)
hub_nodes, hub_elos, non_hub_elos = identify_hubs(G, percentile=95)
print(f"Found {len(hub_nodes)} hubs")
print(f"Average hub Elo: {np.mean(hub_elos):.1f}")
```

### 5. Analyze Ego Networks

```python
from src.ego_network_analysis import build_ego_network, visualize_ego_network

# Build ego network for Magnus Carlsen
G_ego = build_ego_network('carlsen, m.', df_2000s, distance=1)

# Visualize
fig = visualize_ego_network(G_ego)
plt.savefig('results/figures/carlsen_ego_network.png')
```

### 6. Track Player Evolution

```python
from src.ego_network_analysis import analyze_player_evolution

# Analyze player over time
metrics_df, elo_df = analyze_player_evolution(
    'carlsen, m.', 
    df, 
    start_year=2000, 
    end_year=2020
)

# Plot metrics vs Elo
from src.ego_network_analysis import plot_player_metrics
fig = plot_player_metrics(metrics_df, elo_df, metric='Density')
```

### 7. Geographic Analysis

```python
from src.geographic_analysis import (
    analyze_dispersion_by_elo,
    bin_dispersion_by_elo,
    plot_dispersion_vs_elo
)

# Analyze dispersion (requires location data)
dispersion_df = analyze_dispersion_by_elo(df_with_locations, min_games=10)

# Bin by Elo
binned = bin_dispersion_by_elo(dispersion_df, bin_size=50)

# Plot
fig = plot_dispersion_vs_elo(binned)
plt.savefig('results/figures/dispersion_vs_elo.png')
```

### 8. Analyze by Elo Bins

```python
from src.network_construction import analyze_elo_bins

# Analyze networks for different Elo ranges
elo_analysis = analyze_elo_bins(
    df_2000s, 
    players, 
    bin_size=50, 
    elo_min=2000, 
    elo_max=2850
)

# Plot density vs Elo
import matplotlib.pyplot as plt
plt.scatter(elo_analysis['elo_center'], elo_analysis['Densidad'])
plt.xlabel('Elo Rating')
plt.ylabel('Network Density')
plt.show()
```

### 9. Visualize Degree Distribution

```python
from src.network_construction import plot_degree_distribution

# Linear scale
fig1 = plot_degree_distribution(G, log_scale=False)
plt.savefig('results/figures/degree_dist_linear.png')

# Log-log scale with log binning
fig2 = plot_degree_distribution(G, log_scale=True, log_binning=True)
plt.savefig('results/figures/degree_dist_loglog.png')
```

### 10. Calculate Centrality

```python
from src.network_construction import calculate_centrality_measures

# Get centrality for all nodes
centrality_df = calculate_centrality_measures(G)

# Find top 10 by degree centrality
top_10 = centrality_df.nlargest(10, 'DegreeCentrality')
print(top_10[['Player', 'MaxElo', 'DegreeCentrality']])
```

## Key Metrics Explained

### Network Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Density** | E / (N*(N-1)/2) | Fraction of possible edges that exist |
| **Avg Degree** | 2E / N | Average number of connections per player |
| **Clustering** | Triangles / Triples | Tendency to form triangles |
| **Assortativity** | μ in Elo_nn ~ Elo^μ | Tendency to connect with similar Elo |

### Centrality Metrics

| Metric | What it measures | High value means |
|--------|------------------|------------------|
| **Degree** | Number of connections | Well-connected player |
| **Eigenvector** | Connections to important nodes | Connected to important players |

### Geographic Metrics

| Metric | What it measures |
|--------|------------------|
| **Avg Distance** | Mean distance from player's centroid to game locations |
| **Max Distance** | Farthest game from centroid |
| **Total Distance** | Sum of all distances |

## Typical Workflow

```python
# 1. Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocessing import *
from src.network_construction import *

# 2. Load data
df = clean_chess_data('data/raw/chess_games.csv')
df_filtered = filter_by_years(df, 2000, 2020)

# 3. Build network
players = create_player_table(df_filtered)
G = build_chess_network(df_filtered, players)

# 4. Analyze
metrics = analyze_topology(G)
assortativity = calculate_assortativity(G)
centrality_df = calculate_centrality_measures(G)

# 5. Visualize
fig = plot_degree_distribution(G, log_scale=True, log_binning=True)
plt.savefig('results/figures/degree_distribution.png', dpi=300)

# 6. Identify hubs
hubs, hub_elos, non_hub_elos = identify_hubs(G, percentile=95)

# 7. Statistical tests
from scipy.stats import mannwhitneyu
stat, p_value = mannwhitneyu(hub_elos, non_hub_elos)
print(f"Mann-Whitney U test: p = {p_value}")
```

## File Naming Conventions

### Data Files
- `chess_games.csv` - Raw game data
- `worldcities.csv` - Geographic data
- `cleaned_games.csv` - Processed games
- `player_table.csv` - Player statistics

### Result Files
- `degree_distribution.png` - Degree distribution plot
- `assortativity.png` - Assortativity regression
- `centrality_vs_elo.png` - Centrality scatter plots
- `hub_comparison.png` - Hub vs non-hub boxplot
- `density_by_elo.png` - Density across Elo bins
- `dispersion_vs_elo.png` - Geographic dispersion
- `{player}_ego_network.png` - Ego network visualization
- `{player}_evolution.png` - Player evolution over time

## Common Issues and Solutions

### Issue: "Module not found"
```bash
# Make sure you're in the project root directory
cd chess-network-analysis

# Install dependencies
pip install -r requirements.txt
```

### Issue: "File not found"
```python
# Use relative paths from project root
df = clean_chess_data('data/raw/chess_games.csv')  # ✓ Correct
df = clean_chess_data('chess_games.csv')           # ✗ Wrong
```

### Issue: "Network too large to visualize"
```python
# Don't visualize full network (50k+ nodes)
# Instead, visualize ego networks or subgraphs
G_ego = build_ego_network('carlsen, m.', df, distance=1)
visualize_ego_network(G_ego)
```

### Issue: "Out of memory"
```python
# Process data in chunks or filter by year
df_2010 = filter_by_years(df, 2010, 2011)  # Single year
G_2010 = build_chess_network(df_2010, players)
```

## Performance Tips

1. **Use DuckDB for large datasets** - Already implemented in preprocessing
2. **Filter early** - Use `filter_by_years()` before building networks
3. **Cache results** - Save processed data to avoid recomputation
4. **Parallel processing** - Use multiprocessing for independent analyses
5. **Sparse matrices** - NetworkX uses sparse matrices internally

## Next Steps

1. **Explore the code**: Read through `src/` modules
2. **Run examples**: Execute `examples/quick_start.py`
3. **Analyze your data**: Adapt code to your specific questions
4. **Create visualizations**: Generate publication-quality figures
5. **Extend analysis**: Add new metrics or methods

## Resources

- **NetworkX Documentation**: https://networkx.org/documentation/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Network Science Book**: http://networksciencebook.com/
- **Elo Rating System**: https://en.wikipedia.org/wiki/Elo_rating_system

## Getting Help

1. Check function docstrings: `help(function_name)`
2. Read METHODOLOGY.md for algorithm details
3. Review CODE_MAPPING.md to find specific functionality
4. Open an issue on GitHub

## Citation

If you use this code in your research:

```bibtex
@misc{chess_network_analysis_2024,
  author = {Your Name},
  title = {Chess Network Analysis: Elo Rating and Network Topology},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ItzWhole/chess-network-analysis}
}
```
