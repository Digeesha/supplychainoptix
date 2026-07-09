```python
"""
Anomaly detection module using Isolation Forest algorithm.

This module provides functionality for detecting anomalies in supply chain metrics
such as delivery times, order quantities, and transportation delays. It leverages
scikit-learn's Isolation Forest implementation with custom preprocessing and
feature engineering tailored for supply chain data.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for supply chain metrics.
    
    Attributes:
        contamination: Expected proportion of anomalies in the dataset
        n_estimators: Number of base estimators in the ensemble
        max_samples: Number of samples to draw to train each base estimator
        random_state: Random seed for reproducibility
        model: Trained Isolation Forest model
        scaler: StandardScaler for feature normalization
        feature_names: List of feature names used for training
        is_fitted: Boolean indicating if model has been trained
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[int, str] = 'auto',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the anomaly detector with specified parameters.
        
        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
            n_estimators: Number of trees in the forest
            max_samples: Number of samples per tree
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all processors)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            warm_start=False
        )
        
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        
        logger.info(
            f"Initialized AnomalyDetector with contamination={contamination}, "
            f"n_estimators={n_estimators}"
        )
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data format and quality.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            ValueError: If data is invalid or contains issues
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        
        if data.isnull().any().any():
            null_cols = data.columns[data.isnull().any()].tolist()
            logger.warning(f"Data contains null values in columns: {null_cols}")
            raise ValueError(f"Data contains null values in columns: {null_cols}")
        
        if not np.all(np.isfinite(data.select_dtypes(include=[np.number]))):
            raise ValueError("Data contains infinite or NaN values")
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for anomaly detection.
        
        Creates rolling statistics, time-based features, and interaction terms
        that help identify supply chain anomalies.
        
        Args:
            data: Input DataFrame with raw features
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Rolling statistics for time-series features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if len(df) >= 7:
                # 7-day rolling statistics
                df[f'{col}_rolling_mean_7d'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_rolling_std_7d'] = df[col].rolling(window=7, min_periods=1).std()
                
                # Deviation from rolling mean
                df[f'{col}_deviation'] = df[col] - df[f'{col}_rolling_mean_7d']
        
        # Remove any rows with NaN created by feature engineering
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def fit(
        self,
        data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        engineer_features: bool = True
    ) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model on training data.
        
        Args:
            data: Training DataFrame
            feature_cols: List of column names to use as features. If None, uses all numeric columns
            engineer_features: Whether to apply feature engineering
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            logger.info("Starting model training...")
            self._validate_data(data)
            
            # Feature engineering if requested
            if engineer_features:
                logger.info("Engineering features...")
                data = self._engineer_features(data)
            
            # Select features
            if feature_cols is None:
                feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            self.feature_names = feature_cols
            X = data[feature_cols].values
            
            logger.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit the model
            self.model.fit(X_scaled)
            self.is_fitted = True
            
            logger.info("Model training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict(
        self,
        data: pd.DataFrame,
        return_scores: bool = False,
        engineer_features: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict anomalies in the input data.
        
        Args:
            data: DataFrame to predict on
            return_scores: If True, also return anomaly scores
            engineer_features: Whether to apply feature engineering
            
        Returns:
            Array of predictions (1 for normal, -1 for anomaly)
            If return_scores=True, returns tuple of (predictions, scores)
            
        Raises:
            RuntimeError: If model hasn't been fitted
            ValueError: If data validation fails
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            self._validate_data(data)
            
            # Feature engineering if requested
            if engineer_features:
                data = self._engineer_features(data)
            
            # Extract features
            X = data[self.feature_names].values
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            if return_scores:
                # Negative score_samples means higher abnormality
                scores = -self.model.score_samples(X_scaled)
                return predictions, scores
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def get_anomaly_details(
        self,
        data: pd.DataFrame,
        engineer_features: bool = True
    ) -> pd.DataFrame:
        """
        Get detailed anomaly information including scores and probabilities.
        
        Args:
            data: DataFrame to analyze
            engineer_features: Whether to apply feature engineering
            
        Returns:
            DataFrame with original data plus anomaly predictions and scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting anomaly details")
        
        try:
            result_df = data.copy()
            
            predictions, scores = self.predict(
                data,
                return_scores=True,
                engineer_features=engineer_features
            )
            
            result_df['is_anomaly'] = (predictions == -1).astype(int)
            result_df['anomaly_score'] = scores
            
            # Normalize scores to 0-1 range for interpretability
            min_score, max_score = scores.min(), scores.max()
            if max_score > min_score:
                result_df['anomaly_probability'] = (
                    (scores - min_score) / (max_score - min_score)
                )
            else:
                result_df['anomaly_probability'] = 0.5
            
            # Add severity classification
            result_df['severity'] = pd.cut(
                result_df['anomaly_probability'],
                bins=[0, 0.6, 0.8, 1.0],
                labels=['low', 'medium', 'high']
            )
            
            logger.info(
                f"Detected {result_df['is_anomaly'].sum()} anomalies "
                f"out of {len(result_df)} records"
            )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error getting anomaly details: {str(e)}")
            raise
    
    def update(
        self,
        new_data: pd.DataFrame,
        refit: bool = False,
        engineer_features: bool = True
    ) -> 'AnomalyDetector':
        """
        Update the model with new data.
        
        Args:
            new_data: New training data
            refit: If True, completely refit the model. If False, use warm_start
            engineer_features: Whether to apply feature engineering
            
        Returns:
            Self for method chaining
        """
        if not self.is_fitted and not refit:
            logger.warning("Model not fitted yet, forcing refit=True")
            refit = True
        
        try:
            if refit:
                logger.info("Refitting model with new data...")
                return self.fit(new_data, self.feature_names, engineer_features)
            else:
                logger.info("Updating model with new data...")
                self._validate_data(new_data)
                
                if engineer_features:
                    new_data = self._engineer_features(new_data)
                
                X_new = new_data[self.feature_names].values
                X_new_scaled = self.scaler.transform(X_new)
                
                # Create new model with warm_start
                self.model.warm_start = True
                self.model.fit(X_new_scaled)
                self.model.warm_start = False
                
                logger.info("Model update completed")
                return self
                
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and scaler to disk.
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler