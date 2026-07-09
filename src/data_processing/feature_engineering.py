"""
Feature Engineering Module for Supply Chain Disruption Prediction

This module provides comprehensive feature extraction and engineering capabilities
for ML models predicting supply chain disruptions. Features include time-series
decomposition, geopolitical risk scores, logistics metrics, and anomaly indicators.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for supply chain disruption prediction.
    
    Generates temporal, statistical, geopolitical, and logistics-based features
    from raw supply chain data streams.
    """
    
    def __init__(self, 
                 lookback_window: int = 30,
                 scaler_type: str = 'robust',
                 include_interaction_features: bool = True):
        """
        Initialize the FeatureEngineer.
        
        Args:
            lookback_window: Days to look back for rolling statistics
            scaler_type: Type of scaler ('standard' or 'robust')
            include_interaction_features: Whether to generate interaction features
        """
        self.lookback_window = lookback_window
        self.scaler_type = scaler_type
        self.include_interaction_features = include_interaction_features
        
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
        
    def engineer_features(self, 
                         df: pd.DataFrame,
                         fit_scaler: bool = False) -> pd.DataFrame:
        """
        Generate comprehensive feature set from raw data.
        
        Args:
            df: Input DataFrame with raw logistics data
            fit_scaler: Whether to fit the scaler on this data
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature engineering")
            return pd.DataFrame()
        
        logger.info(f"Engineering features for {len(df)} records")
        
        # Create copy to avoid modifying original
        df_features = df.copy()
        
        # Temporal features
        df_features = self._add_temporal_features(df_features)
        
        # Logistics and supply chain specific features
        df_features = self._add_logistics_features(df_features)
        
        # Statistical aggregation features
        df_features = self._add_rolling_statistics(df_features)
        
        # Geopolitical risk features
        df_features = self._add_geopolitical_features(df_features)
        
        # Delay and disruption indicators
        df_features = self._add_disruption_indicators(df_features)
        
        # Seasonality and trend features
        df_features = self._add_seasonality_features(df_features)
        
        # Interaction features
        if self.include_interaction_features:
            df_features = self._add_interaction_features(df_features)
        
        # Handle missing values
        df_features = self._handle_missing_values(df_features)
        
        # Scale numerical features
        df_features = self._scale_features(df_features, fit=fit_scaler)
        
        self.feature_names = [col for col in df_features.columns 
                             if col not in df.columns]
        
        logger.info(f"Generated {len(self.feature_names)} new features")
        return df_features
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from timestamp columns."""
        if 'timestamp' not in df.columns and 'date' not in df.columns:
            logger.warning("No timestamp column found for temporal features")
            return df
        
        time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Basic temporal features
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['day_of_month'] = df[time_col].dt.day
        df['week_of_year'] = df[time_col].dt.isocalendar().week
        df['month'] = df[time_col].dt.month
        df['quarter'] = df[time_col].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df[time_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[time_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[time_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[time_col].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for periodic features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Days since epoch for trend capture
        df['days_since_epoch'] = (df[time_col] - pd.Timestamp('2020-01-01')).dt.days
        
        return df
    
    def _add_logistics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate logistics-specific features."""
        
        # Lead time features
        if 'expected_delivery_date' in df.columns and 'order_date' in df.columns:
            df['planned_lead_time'] = (
                pd.to_datetime(df['expected_delivery_date']) - 
                pd.to_datetime(df['order_date'])
            ).dt.days
        
        if 'actual_delivery_date' in df.columns and 'order_date' in df.columns:
            df['actual_lead_time'] = (
                pd.to_datetime(df['actual_delivery_date']) - 
                pd.to_datetime(df['order_date'])
            ).dt.days
            
            # Lead time variance
            if 'planned_lead_time' in df.columns:
                df['lead_time_variance'] = df['actual_lead_time'] - df['planned_lead_time']
                df['lead_time_variance_pct'] = (
                    df['lead_time_variance'] / df['planned_lead_time'].replace(0, 1)
                ) * 100
        
        # Inventory metrics
        if 'current_inventory' in df.columns and 'average_demand' in df.columns:
            df['days_of_inventory'] = df['current_inventory'] / df['average_demand'].replace(0, 1)
            df['inventory_turnover'] = df['average_demand'] / df['current_inventory'].replace(0, 1)
        
        # Order volume features
        if 'order_quantity' in df.columns:
            df['order_quantity_log'] = np.log1p(df['order_quantity'])
            
        if 'order_value' in df.columns:
            df['order_value_log'] = np.log1p(df['order_value'])
            
            if 'order_quantity' in df.columns:
                df['unit_price'] = df['order_value'] / df['order_quantity'].replace(0, 1)
        
        # Supplier reliability features
        if 'supplier_id' in df.columns:
            supplier_stats = df.groupby('supplier_id').agg({
                'order_id': 'count',
            }).rename(columns={'order_id': 'supplier_order_count'})
            
            df = df.merge(supplier_stats, left_on='supplier_id', 
                         right_index=True, how='left')
        
        # Transportation mode features (if categorical)
        if 'transport_mode' in df.columns:
            transport_dummies = pd.get_dummies(df['transport_mode'], 
                                              prefix='transport_mode')
            df = pd.concat([df, transport_dummies], axis=1)
        
        # Route complexity
        if 'number_of_stops' in df.columns:
            df['route_complexity_score'] = df['number_of_stops']
            
            if 'total_distance_km' in df.columns:
                df['avg_distance_per_stop'] = (
                    df['total_distance_km'] / df['number_of_stops'].replace(0, 1)
                )
        
        return df
    
    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling window statistics for key metrics."""
        if 'timestamp' not in df.columns and 'date' not in df.columns:
            return df
        
        time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df = df.sort_values(time_col)
        
        # Columns to calculate rolling stats for
        rolling_cols = []
        if 'order_quantity' in df.columns:
            rolling_cols.append('order_quantity')
        if 'lead_time_variance' in df.columns:
            rolling_cols.append('lead_time_variance')
        if 'order_value' in df.columns:
            rolling_cols.append('order_value')
        
        windows = [7, 14, 30]  # Weekly, bi-weekly, monthly
        
        for col in rolling_cols:
            for window in windows:
                if window <= len(df):
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}d'] = (
                        df[col].rolling(window=window, min_periods=1).mean()
                    )
                    
                    # Rolling std
                    df[f'{col}_rolling_std_{window}d'] = (
                        df[col].rolling(window=window, min_periods=1).std()
                    )
                    
                    # Rolling min/max
                    df[f'{col}_rolling_min_{window}d'] = (
                        df[col].rolling(window=window, min_periods=1).min()
                    )
                    df[f'{col}_rolling_max_{window}d'] = (
                        df[col].rolling(window=window, min_periods=1).max()
                    )
                    
                    # Coefficient of variation
                    rolling_mean = df[f'{col}_rolling_mean_{window}d']
                    rolling_std = df[f'{col}_rolling_std_{window}d']
                    df[f'{col}_cv_{window}d'] = (
                        rolling_std / rolling_mean.replace(0, np.nan)
                    )
        
        return df
    
    def _add_geopolitical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geopolitical risk and event-based features."""
        
        # Country risk scores (would typically come from external API)
        if 'origin_country' in df.columns:
            # Placeholder for risk scoring - in production, integrate with risk APIs
            df['origin_country_risk_score'] = df['origin_country'].map(
                self._get_country_risk_mapping()
            ).fillna(5)  # Default medium risk
        
        if 'destination_country' in df.columns:
            df['destination_country_risk_score'] = df['destination_country'].map