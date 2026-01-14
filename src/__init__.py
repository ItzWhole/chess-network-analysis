"""
Chess Network Analysis Package

A comprehensive toolkit for analyzing chess player networks based on Elo ratings.

Modules:
--------
- data_preprocessing: Data cleaning and standardization
- network_construction: Network building and topology analysis
- ego_network_analysis: Player-specific network analysis
- geographic_analysis: Geographic dispersion calculations

Author: Matías Laborero
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "Matías Laborero"

from . import data_preprocessing
from . import network_construction
from . import ego_network_analysis
from . import geographic_analysis

__all__ = [
    'data_preprocessing',
    'network_construction',
    'ego_network_analysis',
    'geographic_analysis'
]
