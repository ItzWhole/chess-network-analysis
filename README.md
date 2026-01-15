# Chess Network Analysis: Elo Rating and Network Topology

A comprehensive network analysis investigating the relationship between chess player Elo ratings and network structure in competitive chess games.

## 📊 Project Overview

This project analyzes over 2.7 million chess games (2000-2023) to understand how player skill levels (Elo ratings) influence network topology, connectivity patterns, and geographic dispersion of competitive matches.

### Key Findings

- **Assortativity**: The network exhibits positive assortativity (μ ≈ 0.33) with respect to Elo ratings—players tend to compete against opponents of similar skill levels
- **Hub Behavior**: High-Elo players (hubs) show significantly higher centrality measures, with an "elite club" effect at ratings above 2500
- **Network Density**: Density increases dramatically with Elo (>100x from 2000 to 2800 rating), indicating stronger interconnection among top players
- **Geographic Dispersion**: Higher-rated players travel greater distances for matches, reflecting the global nature of elite competition
- **Scale-Free Properties**: The network follows a power-law degree distribution (γ ≈ 2.3), characteristic of scale-free networks

## 🎯 Research Objectives

1. Evaluate network assortativity with respect to player Elo ratings
2. Analyze the relationship between Elo and centrality measures (degree and eigenvector centrality)
3. Investigate geographic dispersion patterns as a function of player skill level
4. Characterize network evolution through player career progression

## 📁 Repository Structure

```
chess-network-analysis/
├── data/
│   ├── raw/                    # Original chess game data (not included - see Data Sources)
│   └── processed/              # Cleaned and processed datasets
├── src/
│   ├── data_preprocessing.py   # Data cleaning and standardization
│   ├── network_construction.py # Network building and topology analysis
│   ├── ego_network_analysis.py # Player-specific network analysis
│   └── geographic_analysis.py  # Geographic dispersion calculations
├── notebooks/
│   └── analysis.ipynb          # Interactive analysis and visualizations
├── results/
│   ├── figures/                # Generated plots and visualizations
│   └── tables/                 # Statistical results and network metrics
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ItzWhole/chess-network-analysis.git
cd chess-network-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

- Python 3.8+
- NetworkX - Network analysis
- Pandas - Data manipulation
- DuckDB - SQL queries on dataframes
- NumPy - Numerical computations
- Matplotlib - Visualization
- GeoPy - Geographic distance calculations
- SciPy - Statistical analysis

## 🚀 Usage

### 1. Data Preprocessing

```python
from src.data_preprocessing import clean_chess_data, filter_by_years

# Load and clean the raw chess game data
df = clean_chess_data('data/raw/chess_games.csv')

# Filter by time period
df_filtered = filter_by_years(df, 2000, 2020)
```

### 2. Network Construction

```python
from src.network_construction import build_chess_network, analyze_topology

# Build network from games
G = build_chess_network(df_filtered)

# Analyze network properties
metrics = analyze_topology(G)
print(f"Nodes: {metrics['nodes']}, Edges: {metrics['edges']}")
print(f"Average degree: {metrics['avg_degree']:.2f}")
print(f"Clustering coefficient: {metrics['clustering']:.4f}")
```

### 3. Ego Network Analysis

```python
from src.ego_network_analysis import analyze_player_network

# Analyze a specific player's network over time
player_metrics = analyze_player_network(
    player='carlsen, m.',
    start_year=2000,
    end_year=2020,
    distance=1  # First neighbors
)
```

### 4. Geographic Analysis

```python
from src.geographic_analysis import calculate_geographic_dispersion

# Calculate average travel distance for players
dispersion = calculate_geographic_dispersion(df_filtered)
```

## 📈 Key Analyses

### Network Topology (2000-2020)

| Metric | Value |
|--------|-------|
| Nodes | 53,836 |
| Edges | 1,380,116 |
| Average Degree | 51 |
| Network Density | 0.00095 |
| Clustering Coefficient | 0.184 |
| Power-law Exponent (γ) | 2.3 |

### Assortativity Analysis

The network shows positive assortativity with respect to Elo ratings:
- **Slope (μ)**: 0.33
- **R² Score**: 0.865
- **Interpretation**: Players preferentially connect with others of similar skill levels

### Centrality vs. Elo

- **Spearman Correlation (Degree Centrality)**: 0.648
- **Spearman Correlation (Eigenvector Centrality)**: 0.654
- **Hub Definition**: Top 5% by degree centrality
- **Hub Average Elo**: 2506 (vs. 2203 overall)

## 🌍 Geographic Dispersion

Analysis reveals that higher-rated players travel significantly more:
- Players with Elo > 2600 travel ~2-3x farther on average
- Weak but significant correlation between topological and geographic distance (ρ ≈ 0.15, p < 0.0001)

## 📊 Visualizations

The repository includes code to generate:
- Degree distribution plots (linear, log, log-log scales)
- Assortativity regression analysis
- Centrality vs. Elo scatter plots
- Hub comparison boxplots
- Density by Elo bin plots
- Geographic dispersion analysis
- Ego network visualizations for specific players

## 🔬 Methodology

### Data Cleaning

1. **Name Standardization**: Unified player name formats (e.g., "kasparov, g.")
2. **Missing Data**: Removed games without dates or Elo ratings
3. **Outlier Removal**: Filtered unrealistic Elo values (>2882)
4. **Location Mapping**: Matched game locations to countries using geographic databases

### Network Construction

- **Nodes**: Individual players with maximum Elo as attribute
- **Edges**: Games played between players (unweighted, undirected)
- **Time Windows**: Analyzed by year and multi-year periods
- **Elo Bins**: 50-point intervals from 2000-2850

### Statistical Methods

- **Assortativity**: Log-log linear regression of neighbor Elo vs. node Elo
- **Centrality**: Degree and eigenvector centrality measures
- **Correlation**: Spearman rank correlation (non-parametric)
- **Significance Testing**: Mann-Whitney U test for hub comparison

## 📚 Data Sources

- **Chess Games**: [Ajedrez Data](https://www.ajedrezdata.com/) - Over 4.2 million games (filtered to 2.7M)
- **Geographic Data**: [SimpleMaps World Cities Database](https://simplemaps.com/data/world-cities)
- **Country Codes**: ISO 3166-1 alpha-3 standard

## 🎓 Academic Context

This project was developed as part of a network science course, applying complex network theory to real-world competitive chess data. The analysis demonstrates:

- Scale-free network properties in competitive sports
- Homophily effects in skill-based matching
- Geographic constraints on network formation
- Temporal evolution of individual player networks


## 👤 Author

**Matías Laborero**
- GitHub: [@ItzWhole](https://github.com/ItzWhole)
- Project: [chess-network-analysis](https://github.com/ItzWhole/chess-network-analysis)

## 🙏 Acknowledgments

- FIDE (Fédération Internationale des Échecs) for the Elo rating system
- Ajedrez Data for providing comprehensive chess game databases
- NetworkX development team for excellent network analysis tools
