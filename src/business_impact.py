"""
Business Impact Calculator
Berekent potentiële kostenbesparingen en ROI
"""
import pandas as pd
import numpy as np
import config


class BusinessImpactCalculator:
    """Berekent business impact van demand forecasting"""
    
    def __init__(self, forecast_accuracy_improvement=0.3):
        """
        Args:
            forecast_accuracy_improvement: Verwachte verbetering in forecast nauwkeurigheid (0-1)
        """
        self.accuracy_improvement = forecast_accuracy_improvement
        
        # Kosten parameters
        self.storage_cost = config.STORAGE_COST_PER_TON_WEEK
        self.waste_cost = config.WASTE_COST_PER_TON
        self.shortage_cost = config.SHORTAGE_COST_PER_TON
        
        # Huidige situatie
        self.current_waste_pct = config.CURRENT_WASTE_PERCENTAGE / 100
        self.target_waste_pct = config.TARGET_WASTE_PERCENTAGE / 100
        
    def calculate_current_costs(self, total_demand_tons, avg_inventory_weeks=2):
        """Bereken huidige kosten zonder forecasting"""
        # Jaarlijkse vraag
        annual_demand = total_demand_tons
        
        # Verspillingskosten
        waste_tons = annual_demand * self.current_waste_pct
        waste_cost = waste_tons * self.waste_cost
        
        # Opslagkosten (gemiddelde voorraad * weken * kosten)
        avg_inventory = annual_demand / 52 * avg_inventory_weeks
        storage_cost = avg_inventory * avg_inventory_weeks * self.storage_cost
        
        # Shortage kosten (geschat 5% gemiste verkoop)
        shortage_tons = annual_demand * 0.05
        shortage_cost = shortage_tons * self.shortage_cost
        
        total_cost = waste_cost + storage_cost + shortage_cost
        
        return {
            'waste_cost': waste_cost,
            'storage_cost': storage_cost,
            'shortage_cost': shortage_cost,
            'total_cost': total_cost,
            'waste_tons': waste_tons,
            'shortage_tons': shortage_tons
        }
    
    def calculate_optimized_costs(self, total_demand_tons, avg_inventory_weeks=2):
        """Bereken kosten met geoptimaliseerde forecasting"""
        annual_demand = total_demand_tons
        
        # Verbeterde verspilling
        improved_waste_pct = self.current_waste_pct - (
            (self.current_waste_pct - self.target_waste_pct) * self.accuracy_improvement
        )
        waste_tons = annual_demand * improved_waste_pct
        waste_cost = waste_tons * self.waste_cost
        
        # Lagere voorraad door betere planning
        optimized_inventory_weeks = avg_inventory_weeks * (1 - self.accuracy_improvement * 0.3)
        avg_inventory = annual_demand / 52 * optimized_inventory_weeks
        storage_cost = avg_inventory * optimized_inventory_weeks * self.storage_cost
        
        # Minder shortages
        improved_shortage_pct = 0.05 * (1 - self.accuracy_improvement * 0.6)
        shortage_tons = annual_demand * improved_shortage_pct
        shortage_cost = shortage_tons * self.shortage_cost
        
        total_cost = waste_cost + storage_cost + shortage_cost
        
        return {
            'waste_cost': waste_cost,
            'storage_cost': storage_cost,
            'shortage_cost': shortage_cost,
            'total_cost': total_cost,
            'waste_tons': waste_tons,
            'shortage_tons': shortage_tons,
            'waste_pct': improved_waste_pct * 100
        }
    
    def calculate_savings(self, total_demand_tons):
        """Bereken totale besparingen"""
        current = self.calculate_current_costs(total_demand_tons)
        optimized = self.calculate_optimized_costs(total_demand_tons)
        
        savings = {
            'waste_savings': current['waste_cost'] - optimized['waste_cost'],
            'storage_savings': current['storage_cost'] - optimized['storage_cost'],
            'shortage_savings': current['shortage_cost'] - optimized['shortage_cost'],
            'total_savings': current['total_cost'] - optimized['total_cost'],
            'waste_reduction_tons': current['waste_tons'] - optimized['waste_tons'],
            'waste_reduction_pct': ((current['waste_tons'] - optimized['waste_tons']) / 
                                   current['waste_tons'] * 100)
        }
        
        return current, optimized, savings
    
    def calculate_roi(self, total_demand_tons, implementation_cost=50000):
        """Bereken ROI van forecasting systeem"""
        current, optimized, savings = self.calculate_savings(total_demand_tons)
        
        annual_savings = savings['total_savings']
        
        # ROI berekeningen
        payback_period = implementation_cost / annual_savings if annual_savings > 0 else float('inf')
        roi_percentage = (annual_savings / implementation_cost) * 100
        
        # 5-jaar projectie
        year_5_savings = annual_savings * 5
        year_5_roi = ((year_5_savings - implementation_cost) / implementation_cost) * 100
        
        return {
            'implementation_cost': implementation_cost,
            'annual_savings': annual_savings,
            'payback_period_years': payback_period,
            'year_1_roi_pct': roi_percentage,
            'year_5_total_savings': year_5_savings,
            'year_5_roi_pct': year_5_roi,
            'break_even_months': payback_period * 12
        }
    
    def create_impact_summary(self, df):
        """Maak complete business impact samenvatting"""
        # Bereken totale vraag
        total_annual_demand = df['demand_tons'].sum() / 3  # 3 jaar data
        
        # Bereken per product
        product_impacts = []
        
        for product in config.PRODUCTS:
            product_df = df[df['product'] == product]
            product_demand = product_df['demand_tons'].sum() / 3
            
            current, optimized, savings = self.calculate_savings(product_demand)
            
            product_impacts.append({
                'Product': product,
                'Jaarlijkse Vraag (ton)': round(product_demand, 0),
                'Huidige Kosten (EUR)': round(current['total_cost'], 0),
                'Geoptimaliseerde Kosten (EUR)': round(optimized['total_cost'], 0),
                'Jaarlijkse Besparing (EUR)': round(savings['total_savings'], 0),
                'Verspilling Reductie (%)': round(savings['waste_reduction_pct'], 1)
            })
        
        # Totalen
        roi_analysis = self.calculate_roi(total_annual_demand)
        
        return pd.DataFrame(product_impacts), roi_analysis
    
    def generate_impact_report(self, df):
        """Genereer volledige impact report"""
        total_annual_demand = df['demand_tons'].sum() / 3
        
        current, optimized, savings = self.calculate_savings(total_annual_demand)
        roi = self.calculate_roi(total_annual_demand)
        
        report = {
            'Executive Summary': {
                'Totale Jaarlijkse Vraag': f"{total_annual_demand:,.0f} ton",
                'Huidige Jaarlijkse Kosten': f"€{current['total_cost']:,.0f}",
                'Geoptimaliseerde Kosten': f"€{optimized['total_cost']:,.0f}",
                'Jaarlijkse Besparing': f"€{savings['total_savings']:,.0f}",
                'Besparing Percentage': f"{(savings['total_savings']/current['total_cost']*100):.1f}%"
            },
            'Cost Breakdown': {
                'Verspilling Reductie': f"€{savings['waste_savings']:,.0f}",
                'Opslag Optimalisatie': f"€{savings['storage_savings']:,.0f}",
                'Shortage Preventie': f"€{savings['shortage_savings']:,.0f}"
            },
            'Operational Impact': {
                'Verspilling Reductie': f"{savings['waste_reduction_tons']:,.0f} ton ({savings['waste_reduction_pct']:.1f}%)",
                'Huidige Verspilling %': f"{self.current_waste_pct*100:.1f}%",
                'Doel Verspilling %': f"{optimized['waste_pct']:.1f}%"
            },
            'ROI Analysis': {
                'Implementatie Kosten': f"€{roi['implementation_cost']:,.0f}",
                'Break-even Periode': f"{roi['break_even_months']:.1f} maanden",
                'Jaar 1 ROI': f"{roi['year_1_roi_pct']:.1f}%",
                '5-Jaar Besparing': f"€{roi['year_5_total_savings']:,.0f}",
                '5-Jaar ROI': f"{roi['year_5_roi_pct']:.1f}%"
            }
        }
        
        return report


def format_currency(value):
    """Format waarde als EUR currency"""
    return f"€{value:,.2f}"


def format_percentage(value):
    """Format waarde als percentage"""
    return f"{value:.1f}%"


if __name__ == "__main__":
    # Test business impact calculator
    import os
    
    data_path = os.path.join(config.DATA_DIR, 'onion_sales_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['week'])
        
        calculator = BusinessImpactCalculator(forecast_accuracy_improvement=0.3)
        
        # Genereer impact report
        report = calculator.generate_impact_report(df)
        
        print("\n" + "="*60)
        print("BUSINESS IMPACT ANALYSE")
        print("="*60)
        
        for section, data in report.items():
            print(f"\n{section}:")
            print("-" * 40)
            for key, value in data.items():
                print(f"  {key}: {value}")
        
        # Product impacts
        product_df, roi = calculator.create_impact_summary(df)
        print("\n" + "="*60)
        print("IMPACT PER PRODUCT:")
        print("="*60)
        print(product_df.to_string(index=False))
        
    else:
        print("Data file not found. Run data_generator.py first.")
