"""
Quick Start Example for Chess Network Analysis

This script demonstrates the basic workflow for analyzing chess networks.

Usage:
    python examples/quick_start.py
"""

import sys
sys.path.append('..')

from src.data_preprocessing import clean_chess_data, filter_by_years, create_player_table
from src.network_construction import (
    build_chess_network, 
    analyze_topology, 
    calculate_assortativity,
    plot_degree_distribution,
    identify_hubs
)
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Chess Network Analysis - Quick Start")
    print("=" * 60)
    
    # Step 1: Load and clean data
    print("\n[1/5] Loading and cleaning data...")
    print("NOTE: Update the file path to your chess games CSV")
    
    # df = clean_chess_data('data/raw/chess_games.csv')
    # df_filtered = filter_by_years(df, 2000, 2020)
    # print(f"Loaded {len(df_filtered)} games from 2000-2020")
    
    print("(Skipping data load in this example)")
    print("Uncomment the lines above and provide your data file path")
    
    # Step 2: Create player table
    print("\n[2/5] Creating player table...")
    # players_df = create_player_table(df_filtered)
    # print(f"Found {len(players_df)} unique players")
    
    # Step 3: Build network
    print("\n[3/5] Building chess network...")
    # G = build_chess_network(df_filtered, players_df)
    # print(f"Network created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Step 4: Analyze topology
    print("\n[4/5] Analyzing network topology...")
    # metrics = analyze_topology(G)
    # print("\nNetwork Metrics:")
    # print(f"  Nodes: {metrics['nodes']}")
    # print(f"  Edges: {metrics['edges']}")
    # print(f"  Average Degree: {metrics['avg_degree']:.2f}")
    # print(f"  Density: {metrics['density']:.6f}")
    # print(f"  Clustering Coefficient: {metrics['avg_clustering']:.4f}")
    # print(f"  Average Elo: {metrics['avg_elo']:.1f}")
    
    # Step 5: Calculate assortativity
    print("\n[5/5] Calculating assortativity...")
    # assortativity = calculate_assortativity(G)
    # print(f"\nAssortativity Analysis:")
    # print(f"  Coefficient (μ): {assortativity['mu']:.4f}")
    # print(f"  R² Score: {assortativity['r_squared']:.4f}")
    # 
    # if assortativity['mu'] > 0:
    #     print("  → Network is ASSORTATIVE (players connect with similar Elo)")
    # else:
    #     print("  → Network is DISASSORTATIVE")
    
    # Visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # fig1 = plot_degree_distribution(G, log_scale=True, log_binning=True)
    # plt.savefig('results/figures/degree_distribution.png', dpi=300, bbox_inches='tight')
    # print("✓ Saved: results/figures/degree_distribution.png")
    
    # Identify hubs
    # hub_nodes, hub_elos, non_hub_elos = identify_hubs(G, percentile=95)
    # print(f"\n✓ Identified {len(hub_nodes)} hub players (top 5%)")
    # print(f"  Average hub Elo: {np.mean(hub_elos):.1f}")
    # print(f"  Average non-hub Elo: {np.mean(non_hub_elos):.1f}")
    
    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Uncomment the code above and provide your data file")
    print("2. Explore ego network analysis with src/ego_network_analysis.py")
    print("3. Analyze geographic dispersion with src/geographic_analysis.py")
    print("4. Check out the Jupyter notebook in notebooks/analysis.ipynb")


if __name__ == "__main__":
    main()
