"""
Data Processing en Feature Engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config


class DataProcessor:
    """Verwerkt en bereidt data voor voor modeling"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.scalers = {}
        
    def create_lag_features(self, product_df, lag_weeks=[1, 2, 3, 4, 8, 12]):
        """Maak lag features voor tijdreeks voorspelling"""
        for lag in lag_weeks:
            product_df[f'demand_lag_{lag}'] = product_df['demand_tons'].shift(lag)
        
        return product_df
    
    def create_rolling_features(self, product_df, windows=[4, 8, 12]):
        """Maak rolling statistics features"""
        for window in windows:
            product_df[f'demand_rolling_mean_{window}'] = (
                product_df['demand_tons'].rolling(window=window, min_periods=1).mean()
            )
            product_df[f'demand_rolling_std_{window}'] = (
                product_df['demand_tons'].rolling(window=window, min_periods=1).std()
            )
        
        return product_df
    
    def create_date_features(self, product_df):
        """Maak datum-gerelateerde features"""
        # Cyclic encoding voor seizoenaliteit
        product_df['month_sin'] = np.sin(2 * np.pi * product_df['month'] / 12)
        product_df['month_cos'] = np.cos(2 * np.pi * product_df['month'] / 12)
        product_df['week_sin'] = np.sin(2 * np.pi * product_df['week_of_year'] / 52)
        product_df['week_cos'] = np.cos(2 * np.pi * product_df['week_of_year'] / 52)
        
        # Is het hoogseizoen? (Q3/Q4 voor uien)
        product_df['is_high_season'] = product_df['quarter'].isin([3, 4]).astype(int)
        
        return product_df
    
    def create_interaction_features(self, product_df):
        """Maak interactie features"""
        # Prijs * seizoen interactie
        product_df['price_season_interaction'] = (
            product_df['price_eur_per_ton'] * product_df['seasonal_factor']
        )
        
        # Temperatuur * seizoen
        product_df['temp_season_interaction'] = (
            product_df['temperature_avg'] * product_df['is_high_season']
        )
        
        # Markt * prijs
        product_df['market_price_interaction'] = (
            product_df['market_price_index'] * product_df['price_eur_per_ton'] / 1000
        )
        
        return product_df
    
    def prepare_data_for_product(self, product_name):
        """Bereid data voor één product voor"""
        # Filter op product
        product_df = self.df[self.df['product'] == product_name].copy()
        product_df = product_df.sort_values('week').reset_index(drop=True)
        
        # Creëer features
        product_df = self.create_lag_features(product_df)
        product_df = self.create_rolling_features(product_df)
        product_df = self.create_date_features(product_df)
        product_df = self.create_interaction_features(product_df)
        
        # Drop rijen met NaN (door lag features)
        product_df = product_df.dropna()
        
        return product_df
    
    def prepare_all_products(self):
        """Bereid data voor alle producten voor"""
        processed_dfs = []
        
        for product in config.PRODUCTS:
            product_df = self.prepare_data_for_product(product)
            processed_dfs.append(product_df)
        
        return pd.concat(processed_dfs, ignore_index=True)
    
    def get_train_test_split(self, product_df, train_size=config.TRAIN_TEST_SPLIT):
        """Split data in train en test sets"""
        split_idx = int(len(product_df) * train_size)
        
        train_df = product_df.iloc[:split_idx].copy()
        test_df = product_df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def get_feature_columns(self):
        """Return lijst van feature kolommen voor ML modellen"""
        feature_cols = [
            # Lag features
            'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4',
            'demand_lag_8', 'demand_lag_12',
            
            # Rolling features
            'demand_rolling_mean_4', 'demand_rolling_std_4',
            'demand_rolling_mean_8', 'demand_rolling_std_8',
            'demand_rolling_mean_12', 'demand_rolling_std_12',
            
            # Date features
            'month', 'quarter', 'week_of_year',
            'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'is_high_season',
            
            # External features
            'price_eur_per_ton', 'temperature_avg', 'rainfall_mm',
            'market_price_index', 'competition_index', 'consumer_confidence',
            'promotion', 'seasonal_factor', 'trend_factor',
            
            # Interaction features
            'price_season_interaction', 'temp_season_interaction',
            'market_price_interaction'
        ]
        
        return feature_cols


class ProphetDataProcessor:
    """Specifieke data processor voor Facebook Prophet"""
    
    @staticmethod
    def prepare_for_prophet(product_df):
        """Converteer naar Prophet formaat (ds, y kolommen)"""
        prophet_df = pd.DataFrame({
            'ds': product_df['week'],
            'y': product_df['demand_tons']
        })
        
        # Voeg regressors toe
        prophet_df['price'] = product_df['price_eur_per_ton'].values
        prophet_df['temperature'] = product_df['temperature_avg'].values
        prophet_df['rainfall'] = product_df['rainfall_mm'].values
        prophet_df['market_index'] = product_df['market_price_index'].values
        prophet_df['promotion'] = product_df['promotion'].values
        
        return prophet_df
    
    @staticmethod
    def create_future_dataframe(model, periods, last_data):
        """Maak future dataframe met regressors voor Prophet"""
        future = model.make_future_dataframe(periods=periods, freq='W')
        
        # Voeg regressors toe (gebruik laatste bekende waarden als estimate)
        last_values = {
            'price': last_data['price'].iloc[-1],
            'temperature': last_data['temperature'].mean(),  # Gemiddelde als baseline
            'rainfall': last_data['rainfall'].mean(),
            'market_index': last_data['market_index'].iloc[-1],
            'promotion': 0  # Geen promotie in forecast
        }
        
        for col, value in last_values.items():
            future[col] = value
        
        return future


def load_and_process_data(filepath):
    """Laad en verwerk data van CSV"""
    df = pd.read_csv(filepath, parse_dates=['week'])
    processor = DataProcessor(df)
    return processor


if __name__ == "__main__":
    # Test data processing
    import os
    
    data_path = os.path.join(config.DATA_DIR, 'onion_sales_data.csv')
    
    if os.path.exists(data_path):
        processor = load_and_process_data(data_path)
        
        # Test voor één product
        product_df = processor.prepare_data_for_product('Gele_Uien')
        print(f"Processed data shape: {product_df.shape}")
        print(f"\nFeature columns: {processor.get_feature_columns()}")
        print(f"\nSample data:\n{product_df.head()}")
        
        # Train/test split
        train, test = processor.get_train_test_split(product_df)
        print(f"\nTrain size: {len(train)}, Test size: {len(test)}")
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_generator.py first to generate data.")
