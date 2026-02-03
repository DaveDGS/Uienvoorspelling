"""
Data Generator voor realistische uien verkoop data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class OnionDataGenerator:
    """Genereert realistische synthetische verkoop- en voorraaddata"""
    
    def __init__(self, start_date=config.START_DATE, end_date=config.END_DATE):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.weeks = pd.date_range(start=self.start_date, end=self.end_date, freq='W')
        np.random.seed(config.RANDOM_STATE)
        
    def generate_seasonal_pattern(self, product, week_number):
        """Genereer seizoenspatroon voor specifiek product"""
        # Bepaal kwartaal (0-3)
        quarter = (week_number % 52) // 13
        pattern = config.SEASONAL_PATTERNS[product]
        
        # Voeg smoothe transitie toe tussen kwartalen
        quarter_progress = ((week_number % 52) % 13) / 13
        current_factor = pattern[quarter]
        next_factor = pattern[(quarter + 1) % 4]
        
        # Interpoleer tussen kwartalen
        seasonal_factor = current_factor + (next_factor - current_factor) * quarter_progress * 0.3
        
        return seasonal_factor
    
    def generate_weather_data(self):
        """Genereer realistische weersdata"""
        weather_data = []
        
        for i, date in enumerate(self.weeks):
            month = date.month
            
            # Seizoensgebonden temperatuur (Nederland)
            base_temp = {1: 3, 2: 4, 3: 7, 4: 10, 5: 15, 6: 18,
                        7: 20, 8: 20, 9: 16, 10: 12, 11: 7, 12: 4}[month]
            temperature = base_temp + np.random.normal(0, 3)
            
            # Neerslag (mm per week)
            base_rain = {1: 70, 2: 50, 3: 60, 4: 50, 5: 60, 6: 65,
                        7: 75, 8: 80, 9: 75, 10: 80, 11: 85, 12: 75}[month]
            rainfall = max(0, base_rain + np.random.normal(0, 20))
            
            weather_data.append({
                'week': date,
                'temperature_avg': round(temperature, 1),
                'rainfall_mm': round(rainfall, 1)
            })
        
        return pd.DataFrame(weather_data)
    
    def generate_sales_data(self):
        """Genereer verkoop data voor alle producten"""
        sales_data = []
        
        for i, date in enumerate(self.weeks):
            week_number = i
            
            for product in config.PRODUCTS:
                # Basis vraag
                base_demand = config.BASE_DEMAND[product]
                
                # Seizoenspatroon
                seasonal_factor = self.generate_seasonal_pattern(product, week_number)
                
                # Trend (lichte groei over tijd)
                trend_factor = 1 + (week_number / len(self.weeks)) * 0.15
                
                # Prijs effect (hogere prijs = lagere vraag)
                price_min, price_max = config.PRICE_RANGE[product]
                base_price = (price_min + price_max) / 2
                price_variation = np.random.normal(0, (price_max - price_min) * 0.15)
                price = base_price + price_variation
                price_effect = 1 - ((price - base_price) / base_price) * 0.3
                
                # Weers effect (te warm of te koud reduceert vraag)
                month = date.month
                optimal_temp = 15
                temp_deviation = abs(self.generate_temp_for_week(date) - optimal_temp)
                weather_effect = 1 - (temp_deviation / 30) * 0.2
                
                # Random noise
                noise = np.random.normal(1, 0.1)
                
                # Occasionele promoties (5% kans)
                promotion = 1.3 if np.random.random() < 0.05 else 1.0
                
                # Outliers (2% kans op grote afwijking)
                if np.random.random() < 0.02:
                    outlier = np.random.choice([0.5, 1.8])
                else:
                    outlier = 1.0
                
                # Bereken finale vraag
                demand = (base_demand * seasonal_factor * trend_factor * 
                         price_effect * weather_effect * noise * promotion * outlier)
                
                # Rond af en zorg voor positieve waarden
                demand = max(5, round(demand, 1))
                
                sales_data.append({
                    'week': date,
                    'product': product,
                    'demand_tons': demand,
                    'price_eur_per_ton': round(price, 2),
                    'promotion': 1 if promotion > 1 else 0,
                    'seasonal_factor': round(seasonal_factor, 2),
                    'trend_factor': round(trend_factor, 2)
                })
        
        return pd.DataFrame(sales_data)
    
    def generate_temp_for_week(self, date):
        """Helper functie voor temperatuur generatie"""
        month = date.month
        base_temp = {1: 3, 2: 4, 3: 7, 4: 10, 5: 15, 6: 18,
                    7: 20, 8: 20, 9: 16, 10: 12, 11: 7, 12: 4}[month]
        return base_temp + np.random.normal(0, 3)
    
    def generate_market_data(self):
        """Genereer marktprijs indices en economische factoren"""
        market_data = []
        
        for i, date in enumerate(self.weeks):
            # Marktprijs index (100 = baseline)
            base_index = 100
            trend = i * 0.1  # Lichte inflatie
            cycle = 10 * np.sin(2 * np.pi * i / 52)  # Jaarlijkse cyclus
            noise = np.random.normal(0, 5)
            market_index = base_index + trend + cycle + noise
            
            # Concurrentie index
            competition = 100 + np.random.normal(0, 10)
            
            # Consumer confidence
            confidence = 100 + np.random.normal(0, 8)
            
            market_data.append({
                'week': date,
                'market_price_index': round(market_index, 2),
                'competition_index': round(competition, 2),
                'consumer_confidence': round(confidence, 2)
            })
        
        return pd.DataFrame(market_data)
    
    def generate_complete_dataset(self, save_path=None):
        """Genereer complete dataset met alle features"""
        print("Genereren van data...")
        
        # Genereer verschillende data types
        sales_df = self.generate_sales_data()
        weather_df = self.generate_weather_data()
        market_df = self.generate_market_data()
        
        # Merge alle data
        complete_df = sales_df.merge(weather_df, on='week', how='left')
        complete_df = complete_df.merge(market_df, on='week', how='left')
        
        # Voeg extra features toe
        complete_df['year'] = complete_df['week'].dt.year
        complete_df['month'] = complete_df['week'].dt.month
        complete_df['quarter'] = complete_df['week'].dt.quarter
        complete_df['week_of_year'] = complete_df['week'].dt.isocalendar().week
        
        # Sorteer
        complete_df = complete_df.sort_values(['product', 'week']).reset_index(drop=True)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            complete_df.to_csv(save_path, index=False)
            print(f"Data opgeslagen naar: {save_path}")
        
        print(f"Dataset gegenereerd: {len(complete_df)} rijen, {len(config.PRODUCTS)} producten")
        print(f"Periode: {complete_df['week'].min()} tot {complete_df['week'].max()}")
        
        return complete_df


def main():
    """Genereer en sla data op"""
    generator = OnionDataGenerator()
    
    # Genereer data
    df = generator.generate_complete_dataset(
        save_path=os.path.join(config.DATA_DIR, 'onion_sales_data.csv')
    )
    
    # Print statistieken
    print("\n=== DATA STATISTIEKEN ===")
    print(df.groupby('product')['demand_tons'].describe().round(2))
    
    print("\n=== VOORBEELDDATA ===")
    print(df.head(10))


if __name__ == "__main__":
    main()
