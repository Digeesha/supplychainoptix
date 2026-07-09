import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import joblib
import tempfile
import os

from models.anomaly_detector import AnomalyDetector
from models.demand_forecaster import DemandForecaster
from models.disruption_predictor import DisruptionPredictor
from models.model_trainer import ModelTrainer


class TestAnomalyDetector(unittest.TestCase):
    """Test suite for AnomalyDetector model."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.detector = AnomalyDetector(contamination=0.1)
        
        # Create synthetic training data
        np.random.seed(42)
        self.normal_data = pd.DataFrame({
            'lead_time': np.random.normal(10, 2, 100),
            'order_quantity': np.random.normal(1000, 100, 100),
            'supplier_score': np.random.uniform(0.7, 1.0, 100),
            'transit_time': np.random.normal(5, 1, 100)
        })
        
        # Create anomalous data points
        self.anomalous_data = pd.DataFrame({
            'lead_time': [30, 35, 40],
            'order_quantity': [100, 5000, 200],
            'supplier_score': [0.2, 0.1, 0.3],
            'transit_time': [20, 25, 22]
        })

    def test_initialization(self) -> None:
        """Test model initialization with correct parameters."""
        self.assertEqual(self.detector.contamination, 0.1)
        self.assertIsNone(self.detector.model)
        self.assertFalse(self.detector.is_trained)

    def test_fit_valid_data(self) -> None:
        """Test model training with valid data."""
        self.detector.fit(self.normal_data)
        
        self.assertTrue(self.detector.is_trained)
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.feature_columns)
        self.assertEqual(len(self.detector.feature_columns), 4)

    def test_fit_empty_data(self) -> None:
        """Test that fitting with empty data raises ValueError."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.detector.fit(empty_df)
        
        self.assertIn("empty", str(context.exception).lower())

    def test_fit_insufficient_data(self) -> None:
        """Test that fitting with too few samples raises ValueError."""
        small_df = self.normal_data.head(5)
        
        with self.assertRaises(ValueError) as context:
            self.detector.fit(small_df)
        
        self.assertIn("insufficient", str(context.exception).lower())

    def test_predict_anomalies(self) -> None:
        """Test anomaly detection on mixed normal and anomalous data."""
        self.detector.fit(self.normal_data)
        
        test_data = pd.concat([
            self.normal_data.head(10),
            self.anomalous_data
        ], ignore_index=True)
        
        predictions = self.detector.predict(test_data)
        anomaly_scores = self.detector.get_anomaly_scores(test_data)
        
        self.assertEqual(len(predictions), len(test_data))
        self.assertEqual(len(anomaly_scores), len(test_data))
        
        # Expect at least one anomaly detected
        self.assertGreater(np.sum(predictions == -1), 0)
        
        # Anomaly scores for anomalous data should be lower (more negative)
        normal_scores = anomaly_scores[:10]
        anomalous_scores = anomaly_scores[10:]
        self.assertLess(np.mean(anomalous_scores), np.mean(normal_scores))

    def test_predict_before_training(self) -> None:
        """Test that prediction before training raises error."""
        with self.assertRaises(RuntimeError) as context:
            self.detector.predict(self.normal_data)
        
        self.assertIn("not trained", str(context.exception).lower())

    def test_save_and_load_model(self) -> None:
        """Test model serialization and deserialization."""
        self.detector.fit(self.normal_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "anomaly_detector.pkl")
            self.detector.save(model_path)
            
            loaded_detector = AnomalyDetector.load(model_path)
            
            self.assertTrue(loaded_detector.is_trained)
            self.assertEqual(loaded_detector.contamination, self.detector.contamination)
            
            # Verify predictions match
            original_pred = self.detector.predict(self.normal_data.head(20))
            loaded_pred = loaded_detector.predict(self.normal_data.head(20))
            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestDemandForecaster(unittest.TestCase):
    """Test suite for DemandForecaster using Prophet."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.forecaster = DemandForecaster()
        
        # Create synthetic time series data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate trend + seasonality + noise
        trend = np.linspace(1000, 1500, len(dates))
        seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.normal(0, 50, len(dates))
        
        self.train_data = pd.DataFrame({
            'ds': dates,
            'y': trend + seasonal + noise,
            'sku_id': ['SKU_001'] * len(dates)
        })

    def test_initialization(self) -> None:
        """Test forecaster initialization."""
        self.assertIsNone(self.forecaster.model)
        self.assertFalse(self.forecaster.is_trained)
        self.assertEqual(self.forecaster.forecast_horizon, 14)

    def test_fit_valid_data(self) -> None:
        """Test model training with valid time series data."""
        self.forecaster.fit(self.train_data[['ds', 'y']])
        
        self.assertTrue(self.forecaster.is_trained)
        self.assertIsNotNone(self.forecaster.model)

    def test_fit_missing_columns(self) -> None:
        """Test that fitting with missing required columns raises error."""
        invalid_df = pd.DataFrame({
            'date': self.train_data['ds'],
            'value': self.train_data['y']
        })
        
        with self.assertRaises(ValueError) as context:
            self.forecaster.fit(invalid_df)
        
        self.assertIn("ds", str(context.exception).lower())

    def test_forecast_future(self) -> None:
        """Test forecasting future demand."""
        self.forecaster.fit(self.train_data[['ds', 'y']])
        
        forecast_df = self.forecaster.forecast(periods=14)
        
        self.assertEqual(len(forecast_df), 14)
        self.assertIn('ds', forecast_df.columns)
        self.assertIn('yhat', forecast_df.columns)
        self.assertIn('yhat_lower', forecast_df.columns)
        self.assertIn('yhat_upper', forecast_df.columns)
        
        # Verify forecast dates are in the future
        last_train_date = self.train_data['ds'].max()
        first_forecast_date = forecast_df['ds'].min()
        self.assertGreater(first_forecast_date, last_train_date)

    def test_forecast_confidence_intervals(self) -> None:
        """Test that confidence intervals are properly bounded."""
        self.forecaster.fit(self.train_data[['ds', 'y']])
        forecast_df = self.forecaster.forecast(periods=7)
        
        # Lower bound should be less than point estimate
        self.assertTrue((forecast_df['yhat_lower'] <= forecast_df['yhat']).all())
        
        # Upper bound should be greater than point estimate
        self.assertTrue((forecast_df['yhat_upper'] >= forecast_df['yhat']).all())

    def test_forecast_before_training(self) -> None:
        """Test that forecasting before training raises error."""
        with self.assertRaises(RuntimeError) as context:
            self.forecaster.forecast(periods=14)
        
        self.assertIn("not trained", str(context.exception).lower())

    def test_multiple_sku_forecasting(self) -> None:
        """Test forecasting for multiple SKUs."""
        # Create data for multiple SKUs
        sku_data = []
        for sku_id in ['SKU_001', 'SKU_002', 'SKU_003']:
            df = self.train_data.copy()
            df['sku_id'] = sku_id
            df['y'] = df['y'] * np.random.uniform(0.8, 1.2)
            sku_data.append(df)
        
        multi_sku_data = pd.concat(sku_data, ignore_index=True)
        
        forecasts = {}
        for sku in ['SKU_001', 'SKU_002', 'SKU_003']:
            sku_forecaster = DemandForecaster()
            sku_df = multi_sku_data[multi_sku_data['sku_id'] == sku][['ds', 'y']]
            sku_forecaster.fit(sku_df)
            forecasts[sku] = sku_forecaster.forecast(periods=7)
        
        self.assertEqual(len(forecasts), 3)
        for sku, forecast in forecasts.items():
            self.assertEqual(len(forecast), 7)


class TestDisruptionPredictor(unittest.TestCase):
    """Test suite for DisruptionPredictor model."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.predictor = DisruptionPredictor()
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 500
        
        self.train_data = pd.DataFrame({
            'lead_time_variance': np.random.uniform(0, 10, n_samples),
            'supplier_reliability': np.random.uniform(0.5, 1.0, n_samples),
            'geopolitical_risk_score': np.random.uniform(0, 1, n_samples),
            'weather_severity': np.random.uniform(0, 5, n_samples),
            'port_congestion_level': np.random.uniform(0, 10, n_samples),
            'historical_delays': np.random.poisson(2, n_samples),
            'route_complexity': np.random.uniform(1, 10, n_samples)
        })
        
        # Generate labels based on risk factors
        risk_score = (
            self.train_data['lead_time_variance'] * 0.3 +
            (1 - self.train_data['supplier_reliability']) * 0.2 +
            self.train_data['geopolitical_risk_score'] * 0.2 +
            self.train_data['weather_severity'] * 0.15 +