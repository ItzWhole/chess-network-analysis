"""
Geographic Analysis Module

This module analyzes the geographic dispersion of chess games and players.
Includes functions for calculating geodesic distances, finding geographic medians,
and analyzing travel patterns.

Author: Matías Laborero
Date: 2024
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt


def latlon_to_cartesian(lat, lon):
    """
    Convert latitude/longitude to 3D Cartesian coordinates.
    
    Parameters:
    -----------
    lat : float or array
        Latitude in degrees
    lon : float or array
        Longitude in degrees
        
    Returns:
    --------
    numpy.ndarray
        3D Cartesian coordinates (x, y, z)
    """
    lat, lon = np.radians(lat), np.radians(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z]).T


def cartesian_to_latlon(cartesian):
    """
    Convert 3D Cartesian coordinates to latitude/longitude.
    
    Parameters:
    -----------
    cartesian : numpy.ndarray
        3D Cartesian coordinates
        
    Returns:
    --------
    tuple
        (latitude, longitude) in degrees
    """
    x, y, z = cartesian.T
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def geometric_median_sphere(coords, tol=1e-6, max_iter=100):
    """
    Calculate the geometric median on a sphere using Weiszfeld's algorithm.
    
    The geometric median minimizes the sum of geodesic distances to all points.
    This is more robust than the centroid for geographic data.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Array of (latitude, longitude) pairs
    tol : float
        Convergence tolerance (default: 1e-6, ~100 meters)
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    tuple
        (latitude, longitude) of the geometric median
    """
    # Convert to Cartesian coordinates
    cartesian_coords = latlon_to_cartesian(coords[:, 0], coords[:, 1])
    
    # Initial guess: mean of Cartesian coordinates
    guess = np.mean(cartesian_coords, axis=0)
    guess /= np.linalg.norm(guess)  # Normalize to sphere
    
    # Weiszfeld's algorithm
    for iteration in range(max_iter):
        distances = np.linalg.norm(cartesian_coords - guess, axis=1)
        weights = 1 / np.maximum(distances, tol)  # Avoid division by zero
        weighted_sum = np.sum(weights[:, None] * cartesian_coords, axis=0)
        new_guess = weighted_sum / np.linalg.norm(weighted_sum)
        
        # Check convergence
        if np.linalg.norm(new_guess - guess) < tol:
            break
        guess = new_guess
    
    return cartesian_to_latlon(np.array([guess]))


def calculate_player_centroid(player, df):
    """
    Calculate the geographic centroid for a player's games.
    
    Parameters:
    -----------
    player : str
        Player name
    df : pandas.DataFrame
        Games dataframe with CapitalLatitude and CapitalLongitude columns
        
    Returns:
    --------
    tuple
        (latitude, longitude) of the player's geographic centroid
    """
    import duckdb
    
    playergames = duckdb.sql(f'''
    SELECT CapitalLatitude, CapitalLongitude
    FROM df
    WHERE White='{player}' OR Black='{player}'
    ''').to_df().to_numpy()
    
    if len(playergames) == 0:
        return None
    
    centroid = geometric_median_sphere(playergames, tol=1e-6, max_iter=100)
    return centroid


def calculate_dispersion_metrics(player, df):
    """
    Calculate geographic dispersion metrics for a player.
    
    Metrics include:
    - Average distance from centroid
    - Maximum distance from centroid
    - Total distance (sum of all distances)
    - Number of games
    - Player's maximum Elo
    
    Parameters:
    -----------
    player : str
        Player name
    df : pandas.DataFrame
        Games dataframe with location and Elo information
        
    Returns:
    --------
    dict
        Dictionary containing dispersion metrics
    """
    import duckdb
    
    # Get player's game locations
    playergames = duckdb.sql(f'''
    SELECT CapitalLatitude, CapitalLongitude
    FROM df
    WHERE White='{player}' OR Black='{player}'
    ''').to_df().to_numpy()
    
    if len(playergames) == 0:
        return None
    
    # Calculate centroid
    centroid = geometric_median_sphere(playergames, tol=1e-6, max_iter=100)
    
    # Calculate distances
    distances = [geodesic(playergames[i], centroid).kilometers 
                for i in range(len(playergames))]
    
    # Get player's Elo
    elo = duckdb.sql(f'''
    SELECT MAX(GREATEST(WhiteElo, BlackElo)) as MaxElo
    FROM df
    WHERE White='{player}' OR Black='{player}'
    ''').to_df()['MaxElo'].values[0]
    
    return {
        'player': player,
        'avg_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'total_distance': np.sum(distances),
        'num_games': len(playergames),
        'max_elo': int(elo)
    }


def analyze_dispersion_by_elo(df, min_games=10):
    """
    Analyze geographic dispersion as a function of Elo rating.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Games dataframe with location and Elo information
    min_games : int
        Minimum number of games for a player to be included
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with dispersion metrics for all players
    """
    import duckdb
    
    # Get list of players
    players = duckdb.sql('''
    SELECT DISTINCT White AS Player FROM df
    UNION
    SELECT DISTINCT Black AS Player FROM df
    ''').to_df()['Player'].tolist()
    
    # Calculate metrics for each player
    results = []
    for i, player in enumerate(players):
        if i % 100 == 0:
            print(f"Processing player {i}/{len(players)}")
        
        try:
            metrics = calculate_dispersion_metrics(player, df)
            if metrics and metrics['num_games'] >= min_games:
                results.append(metrics)
        except:
            continue
    
    return pd.DataFrame(results)


def bin_dispersion_by_elo(dispersion_df, bin_size=50):
    """
    Bin dispersion metrics by Elo rating.
    
    Parameters:
    -----------
    dispersion_df : pandas.DataFrame
        Dispersion metrics dataframe
    bin_size : int
        Size of Elo bins
        
    Returns:
    --------
    pandas.DataFrame
        Aggregated metrics by Elo bin
    """
    # Create Elo bins
    dispersion_df["elo_bin"] = pd.cut(
        dispersion_df["max_elo"], 
        bins=range(0, dispersion_df["max_elo"].max() + bin_size, bin_size), 
        right=False
    )
    
    dispersion_df["elo_bin_lower"] = dispersion_df["elo_bin"].apply(lambda x: x.left)
    
    # Group by Elo bin and calculate averages
    result = dispersion_df.groupby("elo_bin_lower").agg(
        max_distance_avg=("max_distance", "mean"),
        avg_distance_avg=("avg_distance", "mean"),
        total_distance_avg=("total_distance", "mean"),
        num_players=("player", "count")
    ).reset_index()
    
    return result


def plot_dispersion_vs_elo(binned_df, metric='avg_distance_avg'):
    """
    Plot geographic dispersion against Elo rating.
    
    Parameters:
    -----------
    binned_df : pandas.DataFrame
        Binned dispersion dataframe
    metric : str
        Metric to plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(binned_df['elo_bin_lower'], binned_df[metric], alpha=0.6)
    ax.set_xlabel('Elo Rating')
    ax.set_ylabel('Average Geodesic Distance (km)')
    ax.set_title('Geographic Dispersion vs Elo Rating')
    ax.grid(True, alpha=0.3)
    
    return fig


def analyze_topological_vs_geographic_distance(G, df, num_samples=5000):
    """
    Analyze correlation between topological and geographic distance.
    
    Samples random pairs of nodes and calculates both their shortest path
    length in the network and their geographic distance.
    
    Parameters:
    -----------
    G : networkx.Graph
        Chess player network
    df : pandas.DataFrame
        Games dataframe with location information
    num_samples : int
        Number of random node pairs to sample
        
    Returns:
    --------
    dict
        Dictionary containing:
        - topological_distances: list of shortest path lengths
        - geographic_distances: list of geodesic distances (km)
        - correlations: dict of correlation coefficients
    """
    import random
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    topological_distances = []
    geographic_distances = []
    
    nodes = list(G.nodes())
    
    for i in range(num_samples):
        # Sample two random nodes
        node1, node2 = random.sample(nodes, 2)
        
        # Calculate topological distance
        try:
            topo_dist = nx.shortest_path_length(G, source=node1, target=node2)
        except nx.NetworkXNoPath:
            continue  # Skip if no path exists
        
        # Calculate geographic distance
        centroid1 = calculate_player_centroid(node1, df)
        centroid2 = calculate_player_centroid(node2, df)
        
        if centroid1 is None or centroid2 is None:
            continue
        
        geo_dist = geodesic(centroid1, centroid2).kilometers
        
        topological_distances.append(topo_dist)
        geographic_distances.append(geo_dist)
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(topological_distances, geographic_distances)
    spearman_corr, spearman_p = spearmanr(topological_distances, geographic_distances)
    kendall_corr, kendall_p = kendalltau(topological_distances, geographic_distances)
    
    return {
        'topological_distances': topological_distances,
        'geographic_distances': geographic_distances,
        'correlations': {
            'pearson': {'r': pearson_corr, 'p': pearson_p},
            'spearman': {'rho': spearman_corr, 'p': spearman_p},
            'kendall': {'tau': kendall_corr, 'p': kendall_p}
        }
    }


if __name__ == "__main__":
    print("Geographic Analysis Module")
    print("=" * 50)
    print("\nExample usage:")
    print("dispersion_df = analyze_dispersion_by_elo(df, min_games=10)")
    print("binned_df = bin_dispersion_by_elo(dispersion_df, bin_size=50)")
    print("fig = plot_dispersion_vs_elo(binned_df)")
