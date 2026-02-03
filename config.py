"""
Configuratie voor Onion Demand Forecasting Applicatie
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# Product types
PRODUCTS = [
    'Gele_Uien',
    'Rode_Uien',
    'Sjalotten',
    'Zilveruien',
    'Biologische_Uien'
]

# Data generation parameters
START_DATE = '2021-01-01'
END_DATE = '2024-01-01'
WEEKS = 156  # 3 years of weekly data

# Seizoenspatronen per product (relatieve vraag per kwartaal)
SEASONAL_PATTERNS = {
    'Gele_Uien': [0.7, 0.8, 1.3, 1.2],      # Q1, Q2, Q3, Q4
    'Rode_Uien': [0.8, 0.9, 1.2, 1.1],
    'Sjalotten': [0.9, 0.85, 1.15, 1.1],
    'Zilveruien': [0.75, 0.8, 1.25, 1.2],
    'Biologische_Uien': [1.0, 1.05, 1.1, 0.85]  # Minder seizoensgebonden
}

# Basis vraag per product (tonnen per week)
BASE_DEMAND = {
    'Gele_Uien': 50,
    'Rode_Uien': 30,
    'Sjalotten': 20,
    'Zilveruien': 15,
    'Biologische_Uien': 25
}

# Prijs range per product (EUR per ton)
PRICE_RANGE = {
    'Gele_Uien': (300, 500),
    'Rode_Uien': (400, 650),
    'Sjalotten': (800, 1200),
    'Zilveruien': (350, 550),
    'Biologische_Uien': (600, 900)
}

# Model parameters
TRAIN_TEST_SPLIT = 0.8
FORECAST_HORIZON = 12  # weken
RANDOM_STATE = 42

# Business parameters
STORAGE_COST_PER_TON_WEEK = 5  # EUR
WASTE_COST_PER_TON = 200  # EUR
SHORTAGE_COST_PER_TON = 150  # EUR (gemiste verkoop)
CURRENT_WASTE_PERCENTAGE = 12  # %
TARGET_WASTE_PERCENTAGE = 5   # %

# Visualisatie
PLOTLY_TEMPLATE = 'plotly_white'
COLOR_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
