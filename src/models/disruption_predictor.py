"""
Disruption prediction model for supply chain forecasting.

This module implements a hybrid ML approach combining Prophet time series forecasting
with isolation forest anomaly detection to predict supply chain disruptions 7-14 days ahead.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

logger = logging.getLogger(__name__)


class DisruptionPredictor:
    """
    Hybrid ML model for predicting supply chain disruptions.
    
    Combines Prophet for time series forecasting with Isolation Forest for anomaly detection
    and Random Forest for classification of disruption types.
    """
    
    def __init__(
        self,
        forecast_horizon: int = 14,
        anomaly_contamination: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the disruption predictor.
        
        Args:
            forecast_horizon: Number of days ahead to forecast (7-14 recommended)
            anomaly_contamination: Expected proportion of anomalies in dataset
            random_state: Random seed for reproducibility
        """
        self.forecast_horizon = forecast_horizon
        self.anomaly_contamination = anomaly_contamination
        self.random_state = random_state
        
        # Initialize models
        self.prophet_model: Optional[Prophet] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.disruption_classifier: Optional[RandomForestClassifier] = None
        self.scaler: StandardScaler = StandardScaler()
        
        # Model metadata
        self.feature_columns: List[str] = []
        self.is_trained: bool = False
        self.training_metrics: Dict[str, float] = {}
        
        logger.info(
            f"DisruptionPredictor initialized with forecast_horizon={forecast_horizon} days"
        )
    
    def _prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet time series model.
        
        Args:
            df: DataFrame with datetime index and target variable
            
        Returns:
            DataFrame formatted for Prophet (ds, y columns)
        """
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['lead_time_days'].values
        })
        return prophet_df
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features for ML models.
        
        Features include:
        - Time-based: day_of_week, month, quarter
        - Lagged values: previous 7, 14, 30 days
        - Rolling statistics: mean, std, min, max
        - Rate of change indicators
        - Geopolitical risk scores
        - Port congestion indices
        
        Args:
            df: Raw supply chain data
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Time-based features
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['day_of_year'] = features_df.index.dayofyear
        
        # Lagged features for lead time
        for lag in [1, 3, 7, 14, 30]:
            features_df[f'lead_time_lag_{lag}'] = features_df['lead_time_days'].shift(lag)
        
        # Rolling statistics (7-day window)
        features_df['lead_time_rolling_mean_7'] = (
            features_df['lead_time_days'].rolling(window=7, min_periods=1).mean()
        )
        features_df['lead_time_rolling_std_7'] = (
            features_df['lead_time_days'].rolling(window=7, min_periods=1).std()
        )
        features_df['lead_time_rolling_max_7'] = (
            features_df['lead_time_days'].rolling(window=7, min_periods=1).max()
        )
        
        # Rolling statistics (30-day window)
        features_df['lead_time_rolling_mean_30'] = (
            features_df['lead_time_days'].rolling(window=30, min_periods=1).mean()
        )
        features_df['lead_time_rolling_std_30'] = (
            features_df['lead_time_days'].rolling(window=30, min_periods=1).std()
        )
        
        # Rate of change
        features_df['lead_time_pct_change_7'] = (
            features_df['lead_time_days'].pct_change(periods=7)
        )
        features_df['lead_time_pct_change_14'] = (
            features_df['lead_time_days'].pct_change(periods=14)
        )
        
        # External factors (if available in input data)
        if 'geopolitical_risk_score' in df.columns:
            features_df['geo_risk_ma_7'] = (
                features_df['geopolitical_risk_score'].rolling(window=7, min_periods=1).mean()
            )
        
        if 'port_congestion_index' in df.columns:
            features_df['port_congestion_ma_7'] = (
                features_df['port_congestion_index'].rolling(window=7, min_periods=1).mean()
            )
        
        if 'weather_severity_score' in df.columns:
            features_df['weather_severity_max_7'] = (
                features_df['weather_severity_score'].rolling(window=7, min_periods=1).max()
            )
        
        # Volatility indicators
        features_df['lead_time_volatility'] = (
            features_df['lead_time_days'].rolling(window=14, min_periods=1).std() /
            features_df['lead_time_days'].rolling(window=14, min_periods=1).mean()
        )
        
        # Fill NaN values from rolling calculations
        features_df = features_df.fillna(method='bfill').fillna(method='ffill')
        
        return features_df
    
    def train(
        self,
        train_data: pd.DataFrame,
        target_col: str = 'disruption_occurred',
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train all components of the disruption prediction model.
        
        Args:
            train_data: Historical supply chain data with datetime index
            target_col: Binary target indicating if disruption occurred
            validation_split: Proportion of data to use for validation
            
        Returns:
            Dictionary containing training metrics and model performance
        """
        logger.info("Starting model training...")
        
        if train_data.empty:
            raise ValueError("Training data is empty")
        
        if target_col not in train_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        
        # Extract features
        features_df = self._extract_features(train_data)
        
        # Store feature columns (exclude target and non-feature columns)
        exclude_cols = {target_col, 'lead_time_days', 'order_id', 'supplier_id'}
        self.feature_columns = [
            col for col in features_df.columns 
            if col not in exclude_cols and features_df[col].dtype in ['int64', 'float64']
        ]
        
        logger.info(f"Extracted {len(self.feature_columns)} features for training")
        
        # Train Prophet for time series forecasting
        logger.info("Training Prophet time series model...")
        self.prophet_model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        prophet_data = self._prepare_prophet_data(train_data)
        self.prophet_model.fit(prophet_data)
        
        # Train Isolation Forest for anomaly detection
        logger.info("Training Isolation Forest anomaly detector...")
        X_anomaly = features_df[self.feature_columns].values
        X_anomaly_scaled = self.scaler.fit_transform(X_anomaly)
        
        self.anomaly_detector = IsolationForest(
            contamination=self.anomaly_contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        self.anomaly_detector.fit(X_anomaly_scaled)
        
        # Train Random Forest classifier for disruption prediction
        logger.info("Training Random Forest disruption classifier...")
        X = features_df[self.feature_columns].values
        y = train_data[target_col].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )
        
        self.disruption_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.disruption_classifier.fit(X_train, y_train)
        
        # Evaluate model performance
        y_pred = self.disruption_classifier.predict(X_val)
        y_pred_proba = self.disruption_classifier.predict_proba(X_val)[:, 1]
        
        self.training_metrics = {
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'validation_accuracy': (y_pred == y_val).mean(),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': len(self.feature_columns),
            'trained_at': datetime.now().isoformat()
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.disruption_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.training_metrics['top_features'] = feature_importance.head(10).to_dict('records')
        
        self.is_trained = True
        
        logger.info(
            f"Training complete. ROC-AUC: {self.training_metrics['roc_auc']:.4f}, "
            f"Validation Accuracy: {self.training_metrics['validation_accuracy']:.4f}"
        )
        
        return self.training_metrics
    
    def predict(
        self,
        current_data: pd.DataFrame,
        return_probabilities: bool = True
    ) -> pd.DataFrame:
        """
        Generate disruption predictions for the forecast horizon.
        
        Args:
            current_data: Recent supply chain data (last 30+ days recommended)
            return_probabilities: If True, return probability scores instead of binary predictions
            
        Returns:
            DataFrame with predictions for each day in the forecast horizon
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Generating predictions for