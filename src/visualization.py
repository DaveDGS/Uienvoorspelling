"""
Visualisatie functies voor demand forecasting
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import config


class ForecastVisualizer:
    """Klasse voor het maken van forecast visualisaties"""
    
    def __init__(self):
        self.template = config.PLOTLY_TEMPLATE
        self.colors = config.COLOR_PALETTE
        
    def plot_actual_vs_predicted(self, actual, predicted, dates, model_name, product_name):
        """Plot actuele vs voorspelde waarden"""
        fig = go.Figure()
        
        # Actuele waarden
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines+markers',
            name='Actueel',
            line=dict(color=self.colors[0], width=2),
            marker=dict(size=6)
        ))
        
        # Voorspelde waarden
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted,
            mode='lines+markers',
            name='Voorspeld',
            line=dict(color=self.colors[1], width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'{product_name} - {model_name}: Actueel vs Voorspeld',
            xaxis_title='Week',
            yaxis_title='Vraag (tonnen)',
            template=self.template,
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_forecast_with_confidence(self, forecast_df, product_name):
        """Plot voorspelling met confidence intervals (Prophet)"""
        fig = go.Figure()
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Voorspelling
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines+markers',
            name='Voorspelling',
            line=dict(color=self.colors[1], width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'{product_name} - Vraagvoorspelling (12 weken)',
            xaxis_title='Week',
            yaxis_title='Vraag (tonnen)',
            template=self.template,
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_model_comparison(self, results_dict, product_name):
        """Vergelijk verschillende modellen"""
        models = list(results_dict.keys())
        metrics = ['RMSE', 'MAE', 'MAPE']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=metrics
        )
        
        for idx, metric in enumerate(metrics, 1):
            values = [results_dict[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=self.colors[idx-1],
                    showlegend=False
                ),
                row=1, col=idx
            )
        
        fig.update_layout(
            title=f'{product_name} - Model Vergelijking',
            template=self.template,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_seasonal_patterns(self, df, product_name):
        """Visualiseer seizoenspatronen"""
        product_df = df[df['product'] == product_name].copy()
        
        # Gemiddelde vraag per maand
        monthly_avg = product_df.groupby('month')['demand_tons'].mean().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'],
            y=monthly_avg['demand_tons'],
            marker_color=self.colors,
            text=monthly_avg['demand_tons'].round(1),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'{product_name} - Seizoenspatroon',
            xaxis_title='Maand',
            yaxis_title='Gemiddelde Vraag (tonnen)',
            template=self.template,
            height=450
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df, top_n=15):
        """Plot feature importance"""
        top_features = importance_df.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            marker_color=self.colors[0]
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Belangrijkste Features',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template=self.template,
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_product_comparison(self, df):
        """Vergelijk vraag tussen producten"""
        product_totals = df.groupby('product')['demand_tons'].sum().reset_index()
        product_totals = product_totals.sort_values('demand_tons', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=product_totals['product'],
            x=product_totals['demand_tons'],
            orientation='h',
            marker_color=self.colors,
            text=product_totals['demand_tons'].round(0),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Totale Vraag per Product (3 jaar)',
            xaxis_title='Totale Vraag (tonnen)',
            yaxis_title='Product',
            template=self.template,
            height=400
        )
        
        return fig
    
    def plot_trend_analysis(self, df, product_name):
        """Analyseer trends over tijd"""
        product_df = df[df['product'] == product_name].copy()
        
        # Quarterly aggregation
        quarterly = product_df.groupby(['year', 'quarter']).agg({
            'demand_tons': 'sum',
            'price_eur_per_ton': 'mean'
        }).reset_index()
        
        quarterly['period'] = quarterly['year'].astype(str) + '-Q' + quarterly['quarter'].astype(str)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Vraag Trend', 'Prijs Trend'),
            vertical_spacing=0.15
        )
        
        # Vraag
        fig.add_trace(
            go.Scatter(
                x=quarterly['period'],
                y=quarterly['demand_tons'],
                mode='lines+markers',
                name='Vraag',
                line=dict(color=self.colors[0], width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Prijs
        fig.add_trace(
            go.Scatter(
                x=quarterly['period'],
                y=quarterly['price_eur_per_ton'],
                mode='lines+markers',
                name='Prijs',
                line=dict(color=self.colors[1], width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Periode", row=2, col=1)
        fig.update_yaxes(title_text="Vraag (tonnen)", row=1, col=1)
        fig.update_yaxes(title_text="Prijs (EUR/ton)", row=2, col=1)
        
        fig.update_layout(
            title=f'{product_name} - Trend Analyse',
            template=self.template,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_heatmap(self, df, product_name):
        """Correlatie heatmap voor features"""
        product_df = df[df['product'] == product_name].copy()
        
        # Selecteer numerieke kolommen
        numeric_cols = ['demand_tons', 'price_eur_per_ton', 'temperature_avg', 
                       'rainfall_mm', 'market_price_index', 'seasonal_factor']
        
        corr_data = product_df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlatie")
        ))
        
        fig.update_layout(
            title=f'{product_name} - Feature Correlatie',
            template=self.template,
            height=500,
            width=600
        )
        
        return fig
    
    def create_forecast_table(self, forecast_df, product_name):
        """Maak tabel met voorspellingen"""
        # Format data
        table_df = forecast_df.copy()
        
        if 'ds' in table_df.columns:
            table_df['Week'] = pd.to_datetime(table_df['ds']).dt.strftime('%Y-%m-%d')
            table_df['Voorspelling'] = table_df['yhat'].round(1)
            
            if 'yhat_lower' in table_df.columns:
                table_df['Min'] = table_df['yhat_lower'].round(1)
                table_df['Max'] = table_df['yhat_upper'].round(1)
                display_cols = ['Week', 'Voorspelling', 'Min', 'Max']
            else:
                display_cols = ['Week', 'Voorspelling']
        else:
            display_cols = list(table_df.columns)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=display_cols,
                fill_color=self.colors[0],
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[table_df[col] for col in display_cols],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(
            title=f'{product_name} - Voorspelling Tabel',
            height=400
        )
        
        return fig


def create_summary_statistics(df):
    """Maak samenvatting statistieken"""
    summary = df.groupby('product').agg({
        'demand_tons': ['mean', 'std', 'min', 'max'],
        'price_eur_per_ton': ['mean', 'std']
    }).round(2)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary


if __name__ == "__main__":
    # Test visualisaties
    import os
    
    data_path = os.path.join(config.DATA_DIR, 'onion_sales_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['week'])
        
        viz = ForecastVisualizer()
        
        # Test enkele visualisaties
        fig = viz.plot_seasonal_patterns(df, 'Gele_Uien')
        fig.write_html('/tmp/seasonal_test.html')
        print("Seasonal pattern plot saved to /tmp/seasonal_test.html")
        
        fig = viz.plot_product_comparison(df)
        fig.write_html('/tmp/comparison_test.html')
        print("Product comparison plot saved to /tmp/comparison_test.html")
    else:
        print("Data file not found. Run data_generator.py first.")
