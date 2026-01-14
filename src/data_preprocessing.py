"""
Data Preprocessing Module for Chess Network Analysis

This module handles data cleaning, standardization, and filtering of chess game records.
Includes functions for name normalization, date filtering, and geographic mapping.

Author: Matías Laborero
Date: 2024
"""

import duckdb
import pandas as pd
import re


def reformat_name(name):
    """
    Standardize player names to a consistent format.
    
    Converts names to lowercase and formats as "surname, initial."
    Example: "Garry Kasparov" -> "kasparov, g."
    
    Parameters:
    -----------
    name : str
        Player name in various formats
        
    Returns:
    --------
    str
        Standardized name in format "surname, i."
    """
    # Remove annotations and special characters
    name = name.replace("(wh)", "").replace("(bl)", "")
    name = name.replace("...", "").replace("..", "").replace(".", "")
    name = name.strip().lower()
    
    # Split by comma and strip whitespace
    parts = [part.strip() for part in name.split(",")]
    
    # Ensure we have exactly two parts: "surname" and "given name"
    if len(parts) != 2:
        return name  # Skip malformed names
    
    surnames, given_names = parts
    
    # Extract first surname and first initial
    first_surname = surnames.split()[0]
    first_initial = given_names.split()[0][0]
    
    return f"{first_surname}, {first_initial}."


def clean_chess_data(filepath):
    """
    Load and clean chess game data from CSV file.
    
    Applies multiple filtering steps:
    - Removes games without Elo ratings
    - Filters out online games (chess.com)
    - Removes invalid dates
    - Filters unrealistic Elo values (>2882, <1000)
    - Standardizes player names
    
    Parameters:
    -----------
    filepath : str
        Path to the raw chess games CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned chess games dataframe
    """
    # Load data
    df = pd.read_csv(filepath, on_bad_lines='skip')
    
    # Standardize player names
    df['White'] = df['White'].apply(reformat_name)
    df['Black'] = df['Black'].apply(reformat_name)
    
    # Filter data using DuckDB for efficient SQL-like operations
    filtered_df = duckdb.sql('''
    SELECT *
    FROM df
    WHERE (
        WhiteElo != 'Unknown'
        AND BlackElo != 'Unknown'
        AND Site NOT LIKE '%chess.com%'
        AND Date != '????.??.??'
        AND WhiteElo != '?'
        AND BlackElo != '?'
        AND CAST(WhiteElo AS INTEGER) < 2883 
        AND CAST(BlackElo AS INTEGER) < 2883
        AND CAST(WhiteElo AS INTEGER) > 999 
        AND CAST(BlackElo AS INTEGER) > 999
        AND regexp_matches(TRIM(White), '^[A-Za-zÀ-ÿ-]+, [a-z]\\.$')
        AND regexp_matches(TRIM(Black), '^[A-Za-zÀ-ÿ-]+, [a-z]\\.$')
    );
    ''').to_df()
    
    return filtered_df


def filter_by_years(df, start_year, end_year):
    """
    Filter chess games by date range.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Chess games dataframe with 'Date' column
    start_year : int
        Starting year (inclusive)
    end_year : int
        Ending year (inclusive)
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe containing only games in the specified range
    """
    query = f'''
    SELECT *
    FROM df
    WHERE Date BETWEEN '{start_year}.01.01' AND '{end_year}.12.31'
    ORDER BY Date ASC;
    '''
    
    filtered_df = duckdb.sql(query).to_df()
    return filtered_df


def create_player_table(df):
    """
    Create a table of unique players with their maximum Elo rating.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Chess games dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Player table with columns: Player, MaxElo
    """
    # Combine White and Black players
    playertabletemp = duckdb.sql('''
    SELECT White AS Player, WhiteElo AS Elo
    FROM df
    
    UNION ALL
    
    SELECT Black AS Player, BlackElo AS Elo
    FROM df
    
    ORDER BY Elo ASC
    ''')
    
    # Get maximum Elo for each player
    players = duckdb.sql('''
    SELECT Player, MAX(Elo) AS MaxElo
    FROM playertabletemp
    GROUP BY Player
    ORDER BY MaxElo DESC;
    ''').to_df()
    
    return players


def map_locations_to_countries(df, cities_df):
    """
    Map game locations to countries using a cities database.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Chess games dataframe with 'Site' column
    cities_df : pandas.DataFrame
        Cities database with country information
        
    Returns:
    --------
    pandas.DataFrame
        Games dataframe with added 'country' column
    """
    # Get country abbreviations
    countries_df = get_country_abbreviations()
    
    # Join games with country data
    final_df = duckdb.sql('''
    SELECT a.Date, a.White, a.Black, a.WhiteElo, a.BlackElo, 
           a.Result, a.Site, b.country 
    FROM df AS a
    LEFT JOIN countries_df AS b
    ON a.site LIKE '%' || b.abbreviation || '%'
    WHERE country IS NOT NULL
    ORDER BY RANDOM();
    ''').to_df()
    
    return final_df


def get_country_abbreviations():
    """
    Load country name to abbreviation mappings from CSV file.
    
    This function loads a clean CSV file containing country names, their
    ISO 3166-1 alpha-3 codes, and alternative names/abbreviations commonly
    used in chess databases.
    
    Returns:
    --------
    pandas.DataFrame
        Country mapping with columns: country, abbreviation
        
    Notes:
    ------
    The CSV file includes:
    - Standard country names (e.g., "Germany")
    - ISO 3166-1 alpha-3 codes (e.g., "DEU")
    - Alternative abbreviations used in chess (e.g., "GER")
    - Historical names (e.g., "Swaziland" -> "Eswatini")
    """
    import os
    
    # Try to load from data directory
    csv_path = 'data/country_mappings.csv'
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Expand alternative names into separate rows for matching
        expanded_rows = []
        for _, row in df.iterrows():
            # Add main country name
            expanded_rows.append({
                'country': row['country'],
                'abbreviation': row['abbreviation']
            })
            
            # Add alternative names if they exist
            if pd.notna(row['alternative_names']) and row['alternative_names']:
                alternatives = row['alternative_names'].split('|')
                for alt in alternatives:
                    alt = alt.strip()
                    if alt:
                        expanded_rows.append({
                            'country': row['country'],
                            'abbreviation': alt
                        })
        
        return pd.DataFrame(expanded_rows)
    else:
        # Fallback: return minimal set if CSV not found
        print(f"Warning: {csv_path} not found. Using minimal country set.")
        return _get_minimal_country_set()


def _get_minimal_country_set():
    """
    Fallback function providing a minimal set of country mappings.
    
    This is used only if the country_mappings.csv file is not found.
    For production use, always ensure the CSV file is present.
    
    Returns:
    --------
    pandas.DataFrame
        Minimal country mapping
    """
    data = {
        'country': [
            'Argentina', 'Australia', 'Austria', 'Belgium', 'Brazil', 'Bulgaria',
            'Canada', 'Chile', 'China', 'Croatia', 'Cuba', 'Czech Republic',
            'Denmark', 'England', 'Estonia', 'Finland', 'France', 'Georgia',
            'Germany', 'Greece', 'Hungary', 'Iceland', 'India', 'Iran', 'Ireland',
            'Israel', 'Italy', 'Japan', 'Latvia', 'Lithuania', 'Mexico',
            'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Romania',
            'Russian Federation', 'Scotland', 'Serbia', 'Slovakia', 'Slovenia',
            'South Africa', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine',
            'United Kingdom', 'United States of America', 'Uruguay', 'Vietnam'
        ],
        'abbreviation': [
            'ARG', 'AUS', 'AUT', 'BEL', 'BRA', 'BGR', 'CAN', 'CHL', 'CHN', 'HRV',
            'CUB', 'CZE', 'DNK', 'ENG', 'EST', 'FIN', 'FRA', 'GEO', 'DEU', 'GRC',
            'HUN', 'ISL', 'IND', 'IRN', 'IRL', 'ISR', 'ITA', 'JPN', 'LVA', 'LTU',
            'MEX', 'NLD', 'NZL', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SCO', 'SRB',
            'SVK', 'SVN', 'ZAF', 'ESP', 'SWE', 'CHE', 'TUR', 'UKR', 'GBR', 'USA',
            'URY', 'VNM'
        ]
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    print("Chess Data Preprocessing Module")
    print("================================")
    print("\nExample: Clean and filter chess data")
    print("df = clean_chess_data('data/raw/chess_games.csv')")
    print("df_filtered = filter_by_years(df, 2000, 2020)")
    print("players = create_player_table(df_filtered)")
    print("\nExample: Load country mappings")
    print("countries = get_country_abbreviations()")
    print(f"print(f'Loaded {{len(countries)}} country mappings')")
