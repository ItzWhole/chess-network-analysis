# Methodology and Technical Details

## Overview

This document provides detailed information about the methodology, algorithms, and technical implementation of the chess network analysis project.

## Data Processing Pipeline

### 1. Data Collection

**Source**: Ajedrez Data (https://www.ajedrezdata.com/)
- **Original dataset**: 4.2 million chess games
- **Time period**: Up to April 2023
- **Game format**: Over-the-board (OTB) tournament games

### 2. Data Cleaning

#### Name Standardization
- **Problem**: Player names appear in multiple formats (e.g., "Garry Kasparov", "kasparov, g.", "Kasparov, G")
- **Solution**: Standardized format: `"surname, initial."` (lowercase)
- **Implementation**: Regular expression matching and string manipulation

#### Missing Data Handling
- Removed games without dates
- Removed games without Elo ratings for both players
- Removed online games (chess.com)

#### Outlier Removal
- **Elo range**: 1000-2882 (historical maximum)
- Filtered unrealistic values (>2882 or <1000)

#### Geographic Mapping
- Matched game locations to countries using ISO 3166-1 alpha-3 codes
- Created auxiliary table for country abbreviations
- Used inner join to ensure valid locations
- Assumed games played in capital cities when specific city unknown

**Final dataset**: 2.7 million games (64% of original)

## Network Construction

### Graph Representation

**Type**: Undirected, unweighted graph

**Nodes**:
- Each node represents a unique player
- **Attributes**:
  - `name`: Player name (standardized)
  - `MaxElo`: Maximum Elo rating achieved in the time period

**Edges**:
- Each edge represents at least one game played between two players
- Multiple games between the same pair are collapsed into a single edge
- No self-loops (players don't play themselves)

### Time Windows

Networks were constructed for various time periods:
- **Full period**: 2000-2020 (main analysis)
- **Yearly networks**: Individual years for temporal analysis
- **Elo bins**: 50-point intervals (2000-2050, 2050-2100, ..., 2800-2850)

## Network Analysis Methods

### 1. Degree Distribution

**Power-Law Fitting**:
```
p(k) = C * k^(-γ)
```

Where:
- `k`: node degree
- `p(k)`: probability of degree k
- `γ`: power-law exponent
- `C`: normalization constant

**Method**:
- Log-log linear regression: `log(p(k)) = -γ * log(k) + c`
- Minimum degree threshold: k_min = 94 (determined empirically)
- Result: γ ≈ 2.3 (characteristic of scale-free networks)

### 2. Assortativity Analysis

**Metric**: Correlation between node Elo and average neighbor Elo

**Formula**:
```
Elo_nn(Elo) = a * Elo^μ
```

Where:
- `Elo_nn`: Average Elo of neighbors
- `Elo`: Node's Elo rating
- `μ`: Assortativity coefficient
- `a`: Scaling constant

**Method**:
- Log-log linear regression: `log(Elo_nn) = μ * log(Elo) + log(a)`
- Result: μ ≈ 0.33 (positive assortativity)
- Interpretation: μ > 0 indicates assortative mixing (players connect with similar Elo)

### 3. Centrality Measures

#### Degree Centrality
```
C_D(i) = k_i
```
- Simply the number of connections
- Measures direct connectivity

#### Eigenvector Centrality
```
C_E(i) = (1/λ) * Σ_j A_ij * C_E(j)
```

Where:
- `A_ij`: Adjacency matrix element
- `λ`: Largest eigenvalue of adjacency matrix
- Measures importance based on connections to other important nodes

**Correlation Analysis**:
- Spearman rank correlation (non-parametric)
- Degree centrality vs Elo: ρ = 0.648
- Eigenvector centrality vs Elo: ρ = 0.654

### 4. Hub Identification

**Definition**: Nodes in the top 5% of degree centrality

**Statistical Test**: Mann-Whitney U test
- Null hypothesis: Hub Elo distribution = Non-hub Elo distribution
- Result: p < 0.0001 (highly significant difference)
- Hub average Elo: 2506
- Non-hub average Elo: 2203

### 5. Clustering Coefficient

**Local Clustering**:
```
C_i = (2 * E_i) / (k_i * (k_i - 1))
```

Where:
- `E_i`: Number of edges between neighbors of node i
- `k_i`: Degree of node i

**Global Clustering (Transitivity)**:
```
C_global = (3 * number of triangles) / (number of connected triples)
```

## Geographic Analysis

### 1. Geometric Median Calculation

**Algorithm**: Weiszfeld's algorithm on a sphere

**Steps**:
1. Convert lat/lon to 3D Cartesian coordinates
2. Initialize guess as mean of Cartesian coordinates
3. Iteratively update:
   ```
   x_new = Σ(w_i * x_i) / ||Σ(w_i * x_i)||
   ```
   Where `w_i = 1 / distance(x, x_i)`
4. Project back to sphere surface
5. Converge when angular difference < 0.000057° (~100 meters)

**Purpose**: Find the point that minimizes total geodesic distance to all game locations

### 2. Dispersion Metrics

For each player:
- **Average distance**: Mean geodesic distance from centroid to all game locations
- **Maximum distance**: Farthest game from centroid
- **Total distance**: Sum of all distances

### 3. Topological vs Geographic Distance

**Sampling Method**:
- Random sampling of 50,000 node pairs
- Calculate both topological (shortest path) and geographic (geodesic) distances

**Correlation Measures**:
- Pearson correlation: r = 0.14
- Spearman correlation: ρ = 0.15
- Kendall's tau: τ = 0.11
- All p-values < 0.0001

**Interpretation**: Weak but statistically significant correlation

## Ego Network Analysis

### Construction

**Ego Network**: Subgraph containing:
- Focal player (ego)
- All players within distance d (typically d=1 or d=2)
- All edges between these players

**Node Attributes**:
- `Neighbors`: Distance from ego (0 = ego, 1 = first neighbor, 2 = second neighbor, etc.)

### Temporal Evolution

**Method**:
1. Construct yearly ego networks for a player
2. Calculate topological metrics for each year
3. Track Elo progression over time
4. Correlate network properties with Elo changes

**Key Observations**:
- "Beginner networks" (Elo 2000-2200): Low connectivity, many degree-1 nodes
- "Advanced networks" (Elo > 2200): High connectivity, fewer isolated nodes
- Clustering coefficient increases with Elo up to ~2500

## Statistical Methods

### Correlation Analysis

**Spearman Rank Correlation**:
```
ρ = 1 - (6 * Σd_i²) / (n * (n² - 1))
```

Where:
- `d_i`: Difference in ranks for observation i
- `n`: Number of observations

**Advantages**:
- Non-parametric (no normality assumption)
- Robust to outliers
- Captures monotonic relationships

### Hypothesis Testing

**Mann-Whitney U Test**:
- Non-parametric test for comparing two independent samples
- Used to compare hub vs non-hub Elo distributions
- Null hypothesis: Distributions are identical

## Computational Complexity

### Network Construction
- **Time**: O(N + E) where N = nodes, E = edges
- **Space**: O(N + E)

### Shortest Path Calculations
- **Algorithm**: Breadth-first search (BFS)
- **Time**: O(N + E) per query
- **For all pairs**: O(N * (N + E))

### Centrality Calculations
- **Degree centrality**: O(N + E)
- **Eigenvector centrality**: O(N² * iterations)

### Geographic Median
- **Weiszfeld's algorithm**: O(iterations * num_locations)
- Typical convergence: 10-50 iterations

## Software and Libraries

### Core Libraries
- **NetworkX**: Network construction and analysis
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **DuckDB**: SQL queries on dataframes (faster than pandas for large datasets)

### Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualizations

### Geographic
- **GeoPy**: Geodesic distance calculations
- Uses WGS84 ellipsoid model for Earth

### Statistical
- **SciPy**: Statistical tests and correlations
- **scikit-learn**: Linear regression

## Reproducibility

### Random Seed
- Set random seeds for reproducible sampling
- NetworkX layout algorithms use fixed seeds

### Data Versioning
- Original dataset: Ajedrez Data (April 2023 snapshot)
- Cleaning pipeline documented in code
- Intermediate datasets can be saved for reproducibility

### Computational Environment
- Python 3.8+
- All dependencies specified in requirements.txt
- Platform-independent (tested on Windows, Linux, macOS)

## Limitations and Future Work

### Current Limitations
1. **Temporal aggregation**: Max Elo used instead of time-varying Elo
2. **Location approximation**: Capital cities assumed for many games
3. **Missing data**: ~36% of original games excluded
4. **Online games**: Not included (different dynamics)

### Future Directions
1. **Temporal networks**: Dynamic network analysis with time-varying Elo
2. **Community detection**: Identify clusters of players
3. **Predictive modeling**: Predict match outcomes from network features
4. **Integration**: Combine with online chess platforms (Chess.com, Lichess)
5. **Machine learning**: Network embedding and player similarity

## References

### Elo Rating System
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.

### Network Science
- Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509-512.
- Newman, M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89(20), 208701.

### Algorithms
- Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n points donnés est minimum. *Tohoku Mathematical Journal*, 43, 355-386.

## Contact and Contributions

For questions, suggestions, or contributions, please open an issue on GitHub or contact the author.
