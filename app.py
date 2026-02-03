"""
Streamlit Dashboard voor Onion Demand Forecasting
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from io import BytesIO
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data_processing import DataProcessor, ProphetDataProcessor, load_and_process_data
from src.modeling import ModelTrainer, ForecastModel
from src.visualization import ForecastVisualizer, create_summary_statistics
from src.business_impact import BusinessImpactCalculator

# Page config
st.set_page_config(
    page_title="Uien Vraagvoorspelling Dashboard",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(filepath):
    """Laad data met caching"""
    return pd.read_csv(filepath, parse_dates=['week'])


@st.cache_resource
def load_processor(df):
    """Laad data processor met caching"""
    return DataProcessor(df)


def main():
    """Main dashboard functie"""
    
    # Header
    st.markdown('<div class="main-header">🧅 Uien Vraagvoorspelling Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/2E86AB/FFFFFF?text=Agri+Forecast", 
                 use_column_width=True)
        st.markdown("---")
        
        # Data upload of gebruik standaard data
        st.subheader("📊 Data Selectie")
        
        data_source = st.radio(
            "Kies data bron:",
            ["Gebruik demo data", "Upload eigen data"]
        )
        
        if data_source == "Upload eigen data":
            uploaded_file = st.file_uploader("Upload CSV bestand", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file, parse_dates=['week'])
                st.success("✅ Data geladen!")
            else:
                st.info("Upload een CSV bestand om te beginnen")
                return
        else:
            data_path = os.path.join(config.DATA_DIR, 'onion_sales_data.csv')
            if not os.path.exists(data_path):
                st.error("⚠️ Demo data niet gevonden. Genereer eerst data met data_generator.py")
                return
            df = load_data(data_path)
        
        st.markdown("---")
        
        # Product selectie
        st.subheader("🎯 Product Selectie")
        selected_product = st.selectbox(
            "Kies product:",
            config.PRODUCTS,
            index=0
        )
        
        # Model selectie
        st.subheader("🤖 Model Selectie")
        model_choice = st.selectbox(
            "Kies forecasting model:",
            ["Ensemble", "Prophet", "XGBoost", "Gradient Boosting"],
            index=0
        )
        
        st.markdown("---")
        
        # Forecast parameters
        st.subheader("⚙️ Forecast Parameters")
        forecast_weeks = st.slider(
            "Voorspel aantal weken:",
            min_value=4,
            max_value=26,
            value=12,
            step=1
        )
        
        st.markdown("---")
        
        # Train model button
        if st.button("🚀 Train Modellen", type="primary", use_container_width=True):
            st.session_state['train_models'] = True
    
    # Main content area
    tabs = st.tabs([
        "📈 Overview", 
        "🔮 Voorspelling", 
        "📊 Analyse", 
        "💰 Business Impact",
        "📋 Data"
    ])
    
    # TAB 1: Overview
    with tabs[0]:
        show_overview(df, selected_product)
    
    # TAB 2: Voorspelling
    with tabs[1]:
        show_forecast(df, selected_product, model_choice, forecast_weeks)
    
    # TAB 3: Analyse
    with tabs[2]:
        show_analysis(df, selected_product)
    
    # TAB 4: Business Impact
    with tabs[3]:
        show_business_impact(df)
    
    # TAB 5: Data
    with tabs[4]:
        show_data_tab(df, selected_product)


def show_overview(df, selected_product):
    """Toon overview tab"""
    st.header("📈 Dashboard Overview")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    product_df = df[df['product'] == selected_product]
    
    with col1:
        avg_demand = product_df['demand_tons'].mean()
        st.metric(
            label="Gem. Vraag (ton/week)",
            value=f"{avg_demand:.1f}",
            delta=f"{product_df['demand_tons'].std():.1f} std"
        )
    
    with col2:
        total_demand = product_df['demand_tons'].sum()
        st.metric(
            label="Totale Vraag (3 jaar)",
            value=f"{total_demand:.0f}",
            delta=None
        )
    
    with col3:
        avg_price = product_df['price_eur_per_ton'].mean()
        st.metric(
            label="Gem. Prijs (EUR/ton)",
            value=f"€{avg_price:.0f}",
            delta=f"{product_df['price_eur_per_ton'].std():.0f} std"
        )
    
    with col4:
        promotions = product_df['promotion'].sum()
        st.metric(
            label="Aantal Promoties",
            value=f"{int(promotions)}",
            delta=None
        )
    
    st.markdown("---")
    
    # Visualisaties
    col1, col2 = st.columns(2)
    
    viz = ForecastVisualizer()
    
    with col1:
        st.subheader("Seizoenspatroon")
        fig = viz.plot_seasonal_patterns(df, selected_product)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Vraag Vergelijking per Product")
        fig = viz.plot_product_comparison(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trend analyse
    st.subheader("Trend Analyse")
    fig = viz.plot_trend_analysis(df, selected_product)
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Key Insights")
    
    # Bereken insights
    q3_q4_avg = product_df[product_df['quarter'].isin([3, 4])]['demand_tons'].mean()
    q1_q2_avg = product_df[product_df['quarter'].isin([1, 2])]['demand_tons'].mean()
    seasonal_diff = ((q3_q4_avg - q1_q2_avg) / q1_q2_avg) * 100
    
    st.write(f"""
    - **Seizoenaliteit**: {selected_product} heeft een duidelijk seizoenspatroon met 
      {seasonal_diff:.0f}% hogere vraag in Q3/Q4 (hoogseizoen) vs Q1/Q2
    - **Volatiliteit**: Standaarddeviatie van {product_df['demand_tons'].std():.1f} ton 
      suggereert {'hoge' if product_df['demand_tons'].std() > 10 else 'matige'} vraagvariabiliteit
    - **Prijs-relatie**: Gemiddelde prijs van €{avg_price:.0f}/ton met variatie van 
      €{product_df['price_eur_per_ton'].std():.0f}
    """)
    st.markdown('</div>', unsafe_allow_html=True)


def show_forecast(df, selected_product, model_choice, forecast_weeks):
    """Toon voorspelling tab"""
    st.header("🔮 Vraagvoorspelling")
    
    # Check of modellen getraind moeten worden
    if 'train_models' not in st.session_state:
        st.info("👈 Klik op 'Train Modellen' in de sidebar om te beginnen")
        return
    
    with st.spinner(f"Training {model_choice} model voor {selected_product}..."):
        processor = load_processor(df)
        trainer = ModelTrainer(processor, selected_product)
        
        # Bereid data voor
        product_df, train_df, test_df = trainer.prepare_data()
        
        # Train modellen
        models, results = trainer.train_all_models()
        
        # Selecteer gekozen model
        selected_model = models[model_choice]
    
    st.success(f"✅ {model_choice} model succesvol getraind!")
    
    # Model performance metrics
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", model_choice)
    with col2:
        st.metric("RMSE", f"{results[model_choice]['RMSE']:.2f}")
    with col3:
        st.metric("MAE", f"{results[model_choice]['MAE']:.2f}")
    with col4:
        st.metric("MAPE", f"{results[model_choice]['MAPE']:.1f}%")
    
    st.markdown("---")
    
    # Visualiseer voorspellingen op test set
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Actueel vs Voorspeld (Test Set)")
        
        viz = ForecastVisualizer()
        y_test = test_df['demand_tons'].values
        y_pred = selected_model.predictions
        
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred['yhat'].values
        
        fig = viz.plot_actual_vs_predicted(
            y_test, y_pred, test_df['week'], model_choice, selected_product
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Vergelijking")
        fig = viz.plot_model_comparison(results, selected_product)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Toekomstige voorspellingen
    st.subheader(f"Voorspelling voor komende {forecast_weeks} weken")
    
    if model_choice == "Prophet":
        # Prophet specifieke forecast
        prophet_data = ProphetDataProcessor.prepare_for_prophet(product_df)
        forecast = selected_model.forecast_future(forecast_weeks, prophet_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = viz.plot_forecast_with_confidence(forecast, selected_product)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tabel met voorspellingen
            st.dataframe(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_weeks).style.format({
                    'yhat': '{:.1f}',
                    'yhat_lower': '{:.1f}',
                    'yhat_upper': '{:.1f}'
                }),
                use_container_width=True,
                height=400
            )
    else:
        st.info("Toekomstige voorspellingen zijn momenteel alleen beschikbaar voor Prophet model")
    
    # Download voorspellingen
    st.markdown("---")
    st.subheader("💾 Download Voorspellingen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        if model_choice == "Prophet":
            csv = forecast.to_csv(index=False)
            st.download_button(
                label="📥 Download als CSV",
                data=csv,
                file_name=f"{selected_product}_forecast_{model_choice}.csv",
                mime="text/csv"
            )
    
    with col2:
        # Excel download
        if model_choice == "Prophet":
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                forecast.to_excel(writer, index=False, sheet_name='Forecast')
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download als Excel",
                data=excel_data,
                file_name=f"{selected_product}_forecast_{model_choice}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


def show_analysis(df, selected_product):
    """Toon analyse tab"""
    st.header("📊 Geavanceerde Analyse")
    
    product_df = df[df['product'] == selected_product]
    viz = ForecastVisualizer()
    
    # Feature importance (als XGBoost getraind is)
    if 'train_models' in st.session_state:
        with st.spinner("Training XGBoost voor feature importance..."):
            processor = load_processor(df)
            trainer = ModelTrainer(processor, selected_product)
            models, _ = trainer.train_all_models()
            
            if 'XGBoost' in models:
                xgb_model = models['XGBoost']
                importance_df = xgb_model.get_feature_importance()
                
                st.subheader("🎯 Feature Importance")
                fig = viz.plot_feature_importance(importance_df, top_n=15)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.write(f"""
                **Top 3 belangrijkste factoren voor vraagvoorspelling:**
                1. {importance_df.iloc[0]['feature']}: {importance_df.iloc[0]['importance']:.3f}
                2. {importance_df.iloc[1]['feature']}: {importance_df.iloc[1]['importance']:.3f}
                3. {importance_df.iloc[2]['feature']}: {importance_df.iloc[2]['importance']:.3f}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Correlatie analyse
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Feature Correlaties")
        fig = viz.plot_correlation_heatmap(df, selected_product)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Descriptieve Statistieken")
        stats = product_df[['demand_tons', 'price_eur_per_ton', 'temperature_avg', 
                           'rainfall_mm']].describe()
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
    
    # Distributie analyse
    st.markdown("---")
    st.subheader("📊 Vraag Distributie")
    
    import plotly.figure_factory as ff
    
    fig = ff.create_distplot(
        [product_df['demand_tons'].values],
        ['Vraag'],
        bin_size=5,
        show_rug=False
    )
    fig.update_layout(
        title=f"{selected_product} - Vraag Distributie",
        xaxis_title="Vraag (tonnen)",
        yaxis_title="Dichtheid",
        template=config.PLOTLY_TEMPLATE
    )
    st.plotly_chart(fig, use_container_width=True)


def show_business_impact(df):
    """Toon business impact tab"""
    st.header("💰 Business Impact Analyse")
    
    calculator = BusinessImpactCalculator(forecast_accuracy_improvement=0.3)
    
    # Parameters
    st.subheader("⚙️ Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy_improvement = st.slider(
            "Forecast Verbetering (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Verwachte verbetering in forecast nauwkeurigheid"
        ) / 100
    
    with col2:
        implementation_cost = st.number_input(
            "Implementatie Kosten (EUR)",
            min_value=10000,
            max_value=200000,
            value=50000,
            step=10000
        )
    
    with col3:
        st.metric(
            "Huidige Verspilling",
            f"{config.CURRENT_WASTE_PERCENTAGE}%",
            delta=None
        )
    
    # Update calculator
    calculator.accuracy_improvement = accuracy_improvement
    
    # Bereken impact
    report = calculator.generate_impact_report(df)
    product_impacts, roi = calculator.create_impact_summary(df)
    
    st.markdown("---")
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    summary = report['Executive Summary']
    
    with col1:
        st.metric("Jaarlijkse Besparing", summary['Jaarlijkse Besparing'])
    with col2:
        st.metric("Besparing %", summary['Besparing Percentage'])
    with col3:
        st.metric("Break-even", f"{roi['break_even_months']:.1f} maanden")
    with col4:
        st.metric("5-Jaar ROI", f"{roi['year_5_roi_pct']:.0f}%")
    
    st.markdown("---")
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💵 Cost Breakdown")
        
        cost_data = report['Cost Breakdown']
        cost_df = pd.DataFrame({
            'Categorie': cost_data.keys(),
            'Besparing': [float(v.replace('€', '').replace(',', '')) for v in cost_data.values()]
        })
        
        import plotly.express as px
        fig = px.pie(
            cost_df,
            values='Besparing',
            names='Categorie',
            title='Besparingen per Categorie',
            color_discrete_sequence=config.COLOR_PALETTE
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 ROI Projectie")
        
        # ROI over 5 jaar
        years = list(range(1, 6))
        annual_savings = roi['annual_savings']
        cumulative_savings = [annual_savings * y - implementation_cost for y in years]
        
        import plotly.graph_objects as go
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years,
            y=cumulative_savings,
            name='Cumulatieve Besparing',
            marker_color=config.COLOR_PALETTE[0]
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                     annotation_text="Break-even")
        
        fig.update_layout(
            title="5-Jaar ROI Projectie",
            xaxis_title="Jaar",
            yaxis_title="Cumulatieve Besparing (EUR)",
            template=config.PLOTLY_TEMPLATE
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Impact per product
    st.subheader("📦 Impact per Product")
    st.dataframe(
        product_impacts.style.format({
            'Jaarlijkse Vraag (ton)': '{:.0f}',
            'Huidige Kosten (EUR)': '€{:,.0f}',
            'Geoptimaliseerde Kosten (EUR)': '€{:,.0f}',
            'Jaarlijkse Besparing (EUR)': '€{:,.0f}',
            'Verspilling Reductie (%)': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Operational impact
    st.markdown("---")
    st.subheader("🎯 Operationele Impact")
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    
    operational = report['Operational Impact']
    st.write(f"""
    **Verspilling Reductie:**
    - {operational['Verspilling Reductie']}
    - Van {operational['Huidige Verspilling %']} naar {operational['Doel Verspilling %']}
    
    **Belangrijkste Voordelen:**
    1. Optimalisatie van voorraadniveaus
    2. Vermindering van voedselverspilling
    3. Betere planning en resource allocatie
    4. Verhoogde klanttevredenheid door minder shortages
    5. Duurzamere bedrijfsvoering
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download report
    st.markdown("---")
    
    # Maak Excel report
    from io import BytesIO
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_df = pd.DataFrame([summary]).T
        summary_df.columns = ['Waarde']
        summary_df.to_excel(writer, sheet_name='Executive Summary')
        
        # Product impacts
        product_impacts.to_excel(writer, sheet_name='Product Impact', index=False)
        
        # ROI analysis
        roi_df = pd.DataFrame([roi]).T
        roi_df.columns = ['Waarde']
        roi_df.to_excel(writer, sheet_name='ROI Analyse')
    
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Download Complete Impact Report (Excel)",
        data=excel_data,
        file_name="business_impact_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )


def show_data_tab(df, selected_product):
    """Toon data tab"""
    st.header("📋 Data Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year_filter = st.multiselect(
            "Filter op Jaar",
            options=sorted(df['year'].unique()),
            default=sorted(df['year'].unique())
        )
    
    with col2:
        quarter_filter = st.multiselect(
            "Filter op Kwartaal",
            options=sorted(df['quarter'].unique()),
            default=sorted(df['quarter'].unique())
        )
    
    with col3:
        show_all_products = st.checkbox("Toon alle producten", value=False)
    
    # Filter data
    filtered_df = df[
        (df['year'].isin(year_filter)) &
        (df['quarter'].isin(quarter_filter))
    ]
    
    if not show_all_products:
        filtered_df = filtered_df[filtered_df['product'] == selected_product]
    
    # Statistieken
    st.subheader("📊 Data Statistieken")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Totaal Rijen", len(filtered_df))
    with col2:
        st.metric("Producten", filtered_df['product'].nunique())
    with col3:
        st.metric("Datum Range", f"{filtered_df['week'].min().date()} tot {filtered_df['week'].max().date()}")
    with col4:
        st.metric("Gem. Vraag", f"{filtered_df['demand_tons'].mean():.1f} ton")
    
    st.markdown("---")
    
    # Data tabel
    st.subheader("📄 Data Tabel")
    
    # Selecteer kolommen om te tonen
    display_cols = st.multiselect(
        "Selecteer kolommen",
        options=filtered_df.columns.tolist(),
        default=['week', 'product', 'demand_tons', 'price_eur_per_ton', 
                'temperature_avg', 'market_price_index']
    )
    
    if display_cols:
        st.dataframe(
            filtered_df[display_cols].style.format({
                col: '{:.2f}' for col in display_cols 
                if col in filtered_df.select_dtypes(include=[np.number]).columns
            }),
            use_container_width=True,
            height=400
        )
    
    # Download gefilterde data
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download gefilterde data (CSV)",
            data=csv,
            file_name=f"{selected_product}_data.csv",
            mime="text/csv"
        )
    
    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Data')
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 Download gefilterde data (Excel)",
            data=excel_data,
            file_name=f"{selected_product}_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    
    summary = create_summary_statistics(filtered_df)
    st.dataframe(summary, use_container_width=True)


if __name__ == "__main__":
    main()
