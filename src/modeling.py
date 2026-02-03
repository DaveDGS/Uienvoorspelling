"""
Machine Learning Modellen voor Demand Forecasting
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import pickle
import os
import config

warnings.filterwarnings('ignore')


class ForecastModel:
    """Basis klasse voor forecasting modellen"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.predictions = None
        self.metrics = {}
        
    def calculate_metrics(self, y_true, y_pred):
        """Bereken forecast metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        self.metrics = {
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'MAPE': round(mape, 2)
        }
        
        return self.metrics
    
    def save_model(self, filepath):
        """Sla model op"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filepath):
        """Laad opgeslagen model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ProphetModel(ForecastModel):
    """Facebook Prophet model"""
    
    def __init__(self):
        super().__init__('Prophet')
        
    def train(self, train_df):
        """Train Prophet model"""
        # Initialiseer model met seizoenaliteit
        self.model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Voeg regressors toe
        self.model.add_regressor('price')
        self.model.add_regressor('temperature')
        self.model.add_regressor('rainfall')
        self.model.add_regressor('market_index')
        self.model.add_regressor('promotion')
        
        # Train model
        self.model.fit(train_df)
        
        return self
    
    def predict(self, test_df):
        """Maak voorspellingen"""
        forecast = self.model.predict(test_df)
        self.predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        return forecast['yhat'].values
    
    def forecast_future(self, periods, last_data):
        """Voorspel toekomst"""
        from src.data_processing import ProphetDataProcessor
        
        future = ProphetDataProcessor.create_future_dataframe(
            self.model, periods, last_data
        )
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


class SARIMAXModel(ForecastModel):
    """SARIMAX tijdreeks model"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        super().__init__('SARIMAX')
        self.order = order
        self.seasonal_order = seasonal_order
        
    def train(self, train_df, exog_cols=None):
        """Train SARIMAX model"""
        y = train_df['demand_tons']
        
        # Exogenous variabelen
        exog = None
        if exog_cols:
            exog = train_df[exog_cols]
        
        # Train model
        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False, maxiter=200)
        self.exog_cols = exog_cols
        
        return self
    
    def predict(self, test_df):
        """Maak voorspellingen"""
        exog = None
        if self.exog_cols:
            exog = test_df[self.exog_cols]
        
        forecast = self.fitted_model.forecast(steps=len(test_df), exog=exog)
        self.predictions = forecast
        
        return forecast.values


class XGBoostModel(ForecastModel):
    """XGBoost model voor tijdreeks voorspelling"""
    
    def __init__(self):
        super().__init__('XGBoost')
        
    def train(self, train_df, feature_cols):
        """Train XGBoost model"""
        X_train = train_df[feature_cols]
        y_train = train_df['demand_tons']
        
        # Configureer model
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            objective='reg:squarederror'
        )
        
        # Train
        self.model.fit(X_train, y_train)
        self.feature_cols = feature_cols
        
        return self
    
    def predict(self, test_df):
        """Maak voorspellingen"""
        X_test = test_df[self.feature_cols]
        predictions = self.model.predict(X_test)
        self.predictions = predictions
        
        return predictions
    
    def get_feature_importance(self):
        """Return feature importance"""
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class GradientBoostingModel(ForecastModel):
    """Gradient Boosting model"""
    
    def __init__(self):
        super().__init__('Gradient Boosting')
        
    def train(self, train_df, feature_cols):
        """Train Gradient Boosting model"""
        X_train = train_df[feature_cols]
        y_train = train_df['demand_tons']
        
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=config.RANDOM_STATE
        )
        
        self.model.fit(X_train, y_train)
        self.feature_cols = feature_cols
        
        return self
    
    def predict(self, test_df):
        """Maak voorspellingen"""
        X_test = test_df[self.feature_cols]
        predictions = self.model.predict(X_test)
        self.predictions = predictions
        
        return predictions


class EnsembleModel(ForecastModel):
    """Ensemble model dat meerdere modellen combineert"""
    
    def __init__(self, models, weights=None):
        super().__init__('Ensemble')
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
    def predict(self, test_df):
        """Maak ensemble voorspellingen"""
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predictions') and model.predictions is not None:
                pred = model.predictions
                if isinstance(pred, pd.DataFrame):
                    pred = pred['yhat'].values
                predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        self.predictions = ensemble_pred
        
        return ensemble_pred


class ModelTrainer:
    """Coordinator voor het trainen van alle modellen"""
    
    def __init__(self, processor, product_name):
        self.processor = processor
        self.product_name = product_name
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """Bereid data voor"""
        product_df = self.processor.prepare_data_for_product(self.product_name)
        train_df, test_df = self.processor.get_train_test_split(product_df)
        
        return product_df, train_df, test_df
    
    def train_all_models(self):
        """Train alle modellen"""
        print(f"\n{'='*60}")
        print(f"Training modellen voor: {self.product_name}")
        print(f"{'='*60}")
        
        _, train_df, test_df = self.prepare_data()
        y_test = test_df['demand_tons'].values
        
        # 1. Prophet Model
        print("\n1. Training Prophet model...")
        from src.data_processing import ProphetDataProcessor
        
        prophet_train = ProphetDataProcessor.prepare_for_prophet(train_df)
        prophet_test = ProphetDataProcessor.prepare_for_prophet(test_df)
        
        prophet_model = ProphetModel()
        prophet_model.train(prophet_train)
        prophet_pred = prophet_model.predict(prophet_test)
        prophet_metrics = prophet_model.calculate_metrics(y_test, prophet_pred)
        
        self.models['Prophet'] = prophet_model
        self.results['Prophet'] = prophet_metrics
        print(f"   Prophet - RMSE: {prophet_metrics['RMSE']}, MAE: {prophet_metrics['MAE']}, MAPE: {prophet_metrics['MAPE']}%")
        
        # 2. XGBoost Model
        print("\n2. Training XGBoost model...")
        feature_cols = self.processor.get_feature_columns()
        
        xgb_model = XGBoostModel()
        xgb_model.train(train_df, feature_cols)
        xgb_pred = xgb_model.predict(test_df)
        xgb_metrics = xgb_model.calculate_metrics(y_test, xgb_pred)
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = xgb_metrics
        print(f"   XGBoost - RMSE: {xgb_metrics['RMSE']}, MAE: {xgb_metrics['MAE']}, MAPE: {xgb_metrics['MAPE']}%")
        
        # 3. Gradient Boosting Model
        print("\n3. Training Gradient Boosting model...")
        gb_model = GradientBoostingModel()
        gb_model.train(train_df, feature_cols)
        gb_pred = gb_model.predict(test_df)
        gb_metrics = gb_model.calculate_metrics(y_test, gb_pred)
        
        self.models['Gradient Boosting'] = gb_model
        self.results['Gradient Boosting'] = gb_metrics
        print(f"   Gradient Boosting - RMSE: {gb_metrics['RMSE']}, MAE: {gb_metrics['MAE']}, MAPE: {gb_metrics['MAPE']}%")
        
        # 4. Ensemble Model
        print("\n4. Creating Ensemble model...")
        # Weights gebaseerd op inverse van RMSE
        rmse_values = [prophet_metrics['RMSE'], xgb_metrics['RMSE'], gb_metrics['RMSE']]
        weights = [1/r for r in rmse_values]
        weights = [w/sum(weights) for w in weights]
        
        ensemble_model = EnsembleModel(
            [prophet_model, xgb_model, gb_model],
            weights=weights
        )
        ensemble_pred = ensemble_model.predict(test_df)
        ensemble_metrics = ensemble_model.calculate_metrics(y_test, ensemble_pred)
        
        self.models['Ensemble'] = ensemble_model
        self.results['Ensemble'] = ensemble_metrics
        print(f"   Ensemble - RMSE: {ensemble_metrics['RMSE']}, MAE: {ensemble_metrics['MAE']}, MAPE: {ensemble_metrics['MAPE']}%")
        
        # Bepaal beste model
        best_model = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        print(f"\n{'='*60}")
        print(f"Beste model: {best_model[0]} (RMSE: {best_model[1]['RMSE']})")
        print(f"{'='*60}")
        
        return self.models, self.results
    
    def save_models(self, save_dir):
        """Sla alle modellen op"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(save_dir, f"{self.product_name}_{name}.pkl")
            model.save_model(filepath)
            print(f"Model opgeslagen: {filepath}")
        
        # Sla ook results op
        results_df = pd.DataFrame(self.results).T
        results_path = os.path.join(save_dir, f"{self.product_name}_results.csv")
        results_df.to_csv(results_path)
        print(f"Results opgeslagen: {results_path}")


if __name__ == "__main__":
    # Test modeling
    from src.data_processing import load_and_process_data
    
    data_path = os.path.join(config.DATA_DIR, 'onion_sales_data.csv')
    
    if os.path.exists(data_path):
        processor = load_and_process_data(data_path)
        
        # Train voor één product als test
        trainer = ModelTrainer(processor, 'Gele_Uien')
        models, results = trainer.train_all_models()
        
        # Sla op
        trainer.save_models(config.MODELS_DIR)
    else:
        print("Data file not found. Run data_generator.py first.")
