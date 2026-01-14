"""
Network Construction and Topology Analysis Module

This module builds chess player networks and analyzes their topological properties.
Includes functions for network creation, degree distribution analysis, assortativity,
and centrality measures.

Author: Matías Laborero
Date: 2024
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from sklearn.linear_model import LinearRegression


def build_chess_network(games_df, players_df):
    """
    Construct a chess player network from game records.
    
    Nodes represent players with their maximum Elo rating as an attribute.
    Edges represent games played between players (unweighted, undirected).
    
    Parameters:
    -----------
    games_df : pandas.DataFrame
        Dataframe with columns: White, Black (player names)
    players_df : pandas.DataFrame
        Dataframe with columns: Player, MaxElo
        
    Returns:
    --------
    networkx.Graph
        Undirected graph representing the chess player network
    """
    G = nx.Graph()
    
    # Add nodes with Elo attributes
    for _, row in players_df.iterrows():
        G.add_node(row['Player'], MaxElo=row['MaxElo'], name=row['Player'])
    
    # Add edges from games
    for _, row in games_df.iterrows():
        white_player = row['White']
        black_player = row['Black']
        G.add_edge(white_player, black_player)
    
    return G


def analyze_topology(G):
    """
    Calculate comprehensive topological metrics for a network.
    
    Parameters:
    -----------
    G : networkx.Graph
        Chess player network
        
    Returns:
    --------
    dict
        Dictionary containing network metrics:
        - nodes: number of nodes
        - edges: number of edges
        - avg_degree: average node degree
        - max_degree: maximum node degree
        - min_degree: minimum node degree
        - density: network density
        - avg_clustering: average clustering coefficient
        - global_clustering: global clustering coefficient
        - avg_elo: average Elo rating
        - max_elo: maximum Elo rating
        - min_elo: minimum Elo rating
    """
    degrees = [degree for _, degree in G.degree()]
    elos = [data['MaxElo'] for _, data in G.nodes(data=True)]
    
    metrics = {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'avg_degree': np.mean(degrees),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'global_clustering': nx.transitivity(G),
        'avg_elo': np.mean(elos),
        'max_elo': max(elos),
        'min_elo': min(elos)
    }
    
    return metrics


def plot_degree_distribution(G, log_scale=False, log_binning=False):
    """
    Plot the degree distribution of the network.
    
    Parameters:
    -----------
    G : networkx.Graph
        Chess player network
    log_scale : bool
        If True, use log-log scale
    log_binning : bool
        If True, use logarithmic binning
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    degrees = [degree for _, degree in G.degree()]
    degree_count = Counter(degrees)
    
    if log_binning:
        # Logarithmic binning for better visualization
        max_degree = max(degrees)
        bins = np.logspace(0, np.log10(max_degree), 50)
        hist, bin_edges = np.histogram(degrees, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Remove zero counts
        nonzero = hist > 0
        bin_centers = bin_centers[nonzero]
        hist = hist[nonzero]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(bin_centers, hist, alpha=0.6)
    else:
        degree, count = zip(*sorted(degree_count.items()))
        plt.figure(figsize=(10, 6))
        plt.scatter(degree, count, alpha=0.6)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Degree (log scale)")
        plt.ylabel("Number of Nodes (log scale)")
        plt.title("Log-Log Plot of Degree Distribution")
    else:
        plt.xlabel("Degree")
        plt.ylabel("Number of Nodes")
        plt.title("Degree Distribution")
    
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def fit_power_law(G, k_min=None):
    """
    Fit a power-law distribution to the degree distribution.
    
    Uses linear regression on log-log scale to estimate the exponent γ.
    
    Parameters:
    -----------
    G : networkx.Graph
        Chess player network
    k_min : int, optional
        Minimum degree to consider for fitting
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - gamma: power-law exponent
        - k_min: minimum degree used
        - r_squared: R² score of the fit
    """
    degrees = np.array([degree for _, degree in G.degree()])
    
    if k_min is None:
        k_min = int(np.percentile(degrees, 75))  # Use top 25% of degrees
    
    # Filter degrees >= k_min
    filtered_degrees = degrees[degrees >= k_min]
    
    # Count occurrences
    degree_count = Counter(filtered_degrees)
    k_values = np.array(list(degree_count.keys()))
    counts = np.array(list(degree_count.values()))
    
    # Log-log linear regression
    log_k = np.log10(k_values).reshape(-1, 1)
    log_p = np.log10(counts)
    
    model = LinearRegression()
    model.fit(log_k, log_p)
    
    gamma = -model.coef_[0]
    r_squared = model.score(log_k, log_p)
    
    return {
        'gamma': gamma,
        'k_min': k_min,
        'r_squared': r_squared,
        'intercept': model.intercept_
    }


def calculate_assortativity(G):
    """
    Calculate assortativity with respect to Elo ratings.
    
    Measures the tendency of players to connect with others of similar Elo.
    Uses log-log linear regression: log(Elo_nn) ~ μ * log(Elo) + c
    
    Parameters:
    -----------
    G : networkx.Graph
        Chess player network with MaxElo node attribute
        
    Returns:
    --------
    dict
        Dictionary with keys:
        - mu: assortativity coefficient (slope)
        - intercept: regression intercept
        - r_squared: R² score
    """
    # Calculate average neighbor Elo for each node
    elo_data = []
    
    for node in G.nodes():
        node_elo = G.nodes[node]['MaxElo']
        neighbors = list(G.neighbors(node))
        
        if len(neighbors) > 0:
            neighbor_elos = [G.nodes[n]['MaxElo'] for n in neighbors]
            avg_neighbor_elo = np.mean(neighbor_elos)
            elo_data.append((node_elo, avg_neighbor_elo))
    
    # Convert to arrays
    node_elos = np.array([x[0] for x in elo_data])
    neighbor_elos = np.array([x[1] for x in elo_data])
    
    # Log-log linear regression
    log_node_elo = np.log10(node_elos).reshape(-1, 1)
    log_neighbor_elo = np.log10(neighbor_elos)
    
    model = LinearRegression()
    model.fit(log_node_elo, log_neighbor_elo)
    
    return {
        'mu': model.coef_[0],
        'intercept': model.intercept_,
        'r_squared': model.score(log_node_elo, log_neighbor_elo)
    }


def calculate_centrality_measures(G):
    """
    Calculate degree and eigenvector centrality for all nodes.
    
    Parameters:
    -----------
    G : networkx.Graph
        Chess player network
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with columns: Player, MaxElo, DegreeCentrality, EigenvectorCentrality
    """
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    data = []
    for node in G.nodes():
        data.append({
            'Player': node,
            'MaxElo': G.nodes[node]['MaxElo'],
            'DegreeCentrality': degree_centrality[node],
            'EigenvectorCentrality': eigenvector_centrality[node]
        })
    
    return pd.DataFrame(data)


def identify_hubs(G, percentile=95):
    """
    Identify hub nodes based on degree centrality.
    
    Parameters:
    -----------
    G : networkx.Graph
        Chess player network
    percentile : float
        Percentile threshold for hub identification (default: 95)
        
    Returns:
    --------
    tuple
        (hub_nodes, hub_elos, non_hub_elos)
    """
    degrees = dict(G.degree())
    threshold = np.percentile(list(degrees.values()), percentile)
    
    hub_nodes = [node for node, degree in degrees.items() if degree >= threshold]
    hub_elos = [G.nodes[node]['MaxElo'] for node in hub_nodes]
    non_hub_elos = [G.nodes[node]['MaxElo'] for node in G.nodes() if node not in hub_nodes]
    
    return hub_nodes, hub_elos, non_hub_elos


def analyze_elo_bins(df, players_df, bin_size=50, elo_min=2000, elo_max=2850):
    """
    Analyze network properties for different Elo ranges.
    
    Creates subnetworks for each Elo bin and calculates their properties.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Chess games dataframe
    players_df : pandas.DataFrame
        Players dataframe with MaxElo
    bin_size : int
        Size of Elo bins (default: 50)
    elo_min : int
        Minimum Elo to analyze
    elo_max : int
        Maximum Elo to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with network metrics for each Elo bin
    """
    import duckdb
    
    results = []
    
    for elo_start in range(elo_min, elo_max, bin_size):
        elo_end = elo_start + bin_size
        
        # Filter games where at least one player is in the Elo range
        eloselect = duckdb.sql(f'''
        SELECT DISTINCT * FROM df
        WHERE (WhiteElo > {elo_start} AND WhiteElo < {elo_end}) 
           OR (BlackElo > {elo_start} AND BlackElo < {elo_end})
        ''').to_df()
        
        if len(eloselect) == 0:
            continue
        
        # Build subnetwork
        G_sub = build_chess_network(eloselect, players_df)
        
        if len(G_sub.nodes()) == 0:
            continue
        
        # Calculate metrics
        metrics = analyze_topology(G_sub)
        metrics['elo_range'] = f"{elo_start}-{elo_end}"
        metrics['elo_center'] = (elo_start + elo_end) / 2
        
        results.append(metrics)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Chess Network Construction and Analysis Module")
    print("=" * 50)
    print("\nExample usage:")
    print("G = build_chess_network(games_df, players_df)")
    print("metrics = analyze_topology(G)")
    print("assortativity = calculate_assortativity(G)")
