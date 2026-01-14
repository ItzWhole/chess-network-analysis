"""
Ego Network Analysis Module

This module analyzes player-specific ego networks and their evolution over time.
Includes functions for building ego networks, tracking player progression,
and visualizing network changes.

Author: Matías Laborero
Date: 2024
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb


def build_ego_network(player, df, distance=1):
    """
    Build an ego network for a specific player.
    
    An ego network includes the focal player (ego), their direct neighbors,
    and optionally neighbors at greater distances.
    
    Parameters:
    -----------
    player : str
        Player name (standardized format: "surname, i.")
    df : pandas.DataFrame
        Chess games dataframe
    distance : int
        Network distance to include (1 = direct neighbors, 2 = second neighbors, etc.)
        
    Returns:
    --------
    networkx.Graph
        Ego network with node attribute 'Neighbors' indicating distance from ego
    """
    # Build full network first
    playertabletemp = duckdb.sql('''
    SELECT White AS Player, WhiteElo AS Elo
    FROM df
    UNION ALL
    SELECT Black AS Player, BlackElo AS Elo
    FROM df
    ORDER BY Elo ASC
    ''')
    
    playertable = duckdb.sql('''
    SELECT Player, MAX(Elo) AS MaxElo
    FROM playertabletemp
    GROUP BY Player
    ORDER BY MaxElo ASC;
    ''').to_df()
    
    games_df = duckdb.sql('''
    SELECT White, Black
    FROM df;
    ''').to_df()
    
    # Create full network
    G = nx.Graph()
    for _, row in playertable.iterrows():
        G.add_node(row['Player'], MaxElo=row['MaxElo'], name=row['Player'])
    
    for _, row in games_df.iterrows():
        G.add_edge(row['White'], row['Black'])
    
    # Find neighbors within specified distance
    neighbors_dict = nx.single_source_shortest_path_length(G, player, cutoff=distance)
    nearest_neighbors = pd.DataFrame(list(neighbors_dict.items()), 
                                    columns=['Name', 'Neighbors'])
    
    # Filter games to include only neighbors
    dfplay = duckdb.sql('''
    SELECT *
    FROM df
    WHERE White IN (SELECT Name FROM nearest_neighbors)
       OR Black IN (SELECT Name FROM nearest_neighbors)
    ''').to_df()
    
    # Build ego network
    playertabletemp = duckdb.sql('''
    SELECT White AS Player, WhiteElo AS Elo
    FROM dfplay
    UNION ALL
    SELECT Black AS Player, BlackElo AS Elo
    FROM dfplay
    ORDER BY Elo ASC
    ''')
    
    playerframe = duckdb.sql('''
    SELECT p.Player, MAX(p.Elo) AS MaxElo, k.Neighbors
    FROM playertabletemp AS p
    LEFT OUTER JOIN nearest_neighbors AS k
    ON p.Player = k.Name
    GROUP BY p.Player, k.Neighbors
    ORDER BY MaxElo ASC;
    ''').to_df()
    
    # Mark nodes not in ego network with distance 1000
    playerframe['Neighbors'] = playerframe['Neighbors'].fillna(1000)
    
    gamesframe = duckdb.sql('''
    SELECT White, Black
    FROM dfplay;
    ''').to_df()
    
    # Create ego network graph
    G_ego = nx.Graph()
    for _, row in playerframe.iterrows():
        G_ego.add_node(row['Player'], 
                      MaxElo=row['MaxElo'], 
                      name=row['Player'], 
                      Neighbors=row['Neighbors'])
    
    for _, row in gamesframe.iterrows():
        G_ego.add_edge(row['White'], row['Black'])
    
    return G_ego


def visualize_ego_network(G_ego, figsize=(12, 8)):
    """
    Visualize an ego network with color-coded distance levels.
    
    Parameters:
    -----------
    G_ego : networkx.Graph
        Ego network with 'Neighbors' node attribute
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the visualization
    """
    # Color mapping based on distance
    color_map = []
    size_map = []
    
    for node in G_ego.nodes:
        distance = G_ego.nodes[node]["Neighbors"]
        
        if distance == 0:  # Ego node
            color_map.append("red")
            size_map.append(30)
        elif distance == 1:  # First neighbors
            color_map.append("blue")
            size_map.append(15)
        elif distance == 1000:  # Second neighbors (not directly connected)
            color_map.append("gray")
            size_map.append(2)
        else:
            color_map.append("black")
            size_map.append(5)
    
    # Create layout and draw
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G_ego, k=0.5, iterations=50)
    nx.draw(G_ego, pos, 
            with_labels=False, 
            node_size=size_map, 
            node_color=color_map, 
            width=0.1, 
            alpha=0.7)
    
    plt.title("Ego Network Visualization")
    plt.axis('off')
    
    return plt.gcf()


def analyze_player_evolution(player, df, start_year, end_year):
    """
    Analyze how a player's ego network evolves over time.
    
    Creates yearly ego networks and tracks topological changes.
    
    Parameters:
    -----------
    player : str
        Player name
    df : pandas.DataFrame
        Full chess games dataframe
    start_year : int
        Starting year for analysis
    end_year : int
        Ending year for analysis
        
    Returns:
    --------
    tuple
        (metrics_df, elo_df) - Network metrics and Elo progression dataframes
    """
    from src.data_preprocessing import filter_by_years
    
    # Get player's Elo progression
    dfplay = duckdb.sql(f'''
    SELECT *
    FROM df
    WHERE White = '{player}' OR Black = '{player}'
    ORDER BY Date ASC
    ''').to_df()
    
    playerelo = duckdb.sql(f'''
    SELECT WhiteElo AS elo, Date FROM dfplay
    WHERE White='{player}'
    UNION 
    SELECT BlackElo AS elo, Date FROM dfplay
    WHERE Black='{player}'
    ORDER BY Date ASC
    ''').to_df()
    
    playerelofinal = duckdb.sql('''
    SELECT 
        EXTRACT(YEAR FROM STRPTIME(Date, '%Y.%m.%d')) AS year,
        MAX(elo) AS max_elo
    FROM playerelo
    WHERE Date NOT LIKE '%.??.%' AND Date NOT LIKE '%.%.??'
    GROUP BY year
    ORDER BY year;
    ''').to_df()
    
    # Build yearly ego networks
    networks = []
    network_names = []
    
    for year in range(start_year, end_year):
        df_year = filter_by_years(df, year, year + 1)
        try:
            G_ego = build_ego_network(player, df_year, distance=1)
            networks.append(G_ego)
            network_names.append(f"{player} {year}-{year+1}")
        except:
            # Player may not have played in this year
            continue
    
    # Calculate metrics for each network
    metrics_data = {
        'Network': network_names,
        'Nodes': [len(G.nodes()) for G in networks],
        'Edges': [len(G.edges()) for G in networks],
        'AvgDegree': [np.mean([degree for _, degree in G.degree()]) for G in networks],
        'MaxDegree': [max([degree for _, degree in G.degree()]) for G in networks],
        'MinDegree': [min([degree for _, degree in G.degree()]) for G in networks],
        'Density': [nx.density(G) for G in networks],
        'AvgClustering': [nx.average_clustering(G) for G in networks],
        'GlobalClustering': [nx.transitivity(G) for G in networks]
    }
    
    # Add diameter (only for connected component)
    diameters = []
    for G in networks:
        if nx.is_connected(G):
            diameters.append(nx.diameter(G))
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            diameters.append(nx.diameter(G.subgraph(largest_cc)))
    metrics_data['Diameter'] = diameters
    
    # Calculate fraction of degree-1 nodes
    fraction_degree_1 = [
        sum(1 for _, degree in G.degree() if degree == 1) / len(G.nodes()) 
        if len(G.nodes()) > 0 else 0
        for G in networks
    ]
    metrics_data['FractionDegree1'] = fraction_degree_1
    
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df, playerelofinal


def compare_player_networks(players, df, year_start, year_end):
    """
    Compare ego network evolution for multiple players.
    
    Parameters:
    -----------
    players : list
        List of player names
    df : pandas.DataFrame
        Chess games dataframe
    year_start : int
        Starting year
    year_end : int
        Ending year
        
    Returns:
    --------
    dict
        Dictionary mapping player names to (metrics_df, elo_df) tuples
    """
    results = {}
    
    for player in players:
        try:
            metrics_df, elo_df = analyze_player_evolution(
                player, df, year_start, year_end
            )
            results[player] = (metrics_df, elo_df)
        except Exception as e:
            print(f"Error analyzing {player}: {e}")
            continue
    
    return results


def plot_player_metrics(metrics_df, elo_df, metric='Density'):
    """
    Plot a network metric against Elo rating over time.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        Network metrics dataframe
    elo_df : pandas.DataFrame
        Elo progression dataframe
    metric : str
        Metric to plot (column name from metrics_df)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Merge dataframes
    df_combined = pd.concat([metrics_df, elo_df], axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_combined['max_elo'], df_combined[metric], alpha=0.6)
    ax.set_xlabel('Elo Rating')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Elo Rating')
    ax.grid(True, alpha=0.3)
    
    return fig


if __name__ == "__main__":
    print("Ego Network Analysis Module")
    print("=" * 50)
    print("\nExample usage:")
    print("G_ego = build_ego_network('carlsen, m.', df, distance=1)")
    print("metrics_df, elo_df = analyze_player_evolution('carlsen, m.', df, 2000, 2020)")
