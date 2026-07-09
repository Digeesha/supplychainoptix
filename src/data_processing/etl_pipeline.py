"""
ETL Pipeline for Supply Chain Data Ingestion and Transformation

This module handles extraction, transformation, and loading of multi-source
logistics data including shipments, inventory levels, supplier performance,
and geopolitical events.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config.settings import (
    DATABASE_URL,
    API_ENDPOINTS,
    DATA_RETENTION_DAYS,
    ETL_BATCH_SIZE
)
from ..utils.validators import validate_shipment_data, validate_inventory_data
from ..utils.transformers import normalize_location_data, calculate_transit_metrics


logger = logging.getLogger(__name__)


@dataclass
class ETLMetrics:
    """Tracks ETL pipeline execution metrics"""
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    records_failed: int = 0
    execution_time_seconds: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DataSourceConnector:
    """Manages connections to external data sources with retry logic"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.session = self._create_session(max_retries)
    
    def _create_session(self, max_retries: int) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def fetch_data(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch data from external API endpoint"""
        try:
            response = self.session.get(
                endpoint,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from {endpoint}: {str(e)}")
            return None


class ETLPipeline:
    """
    Main ETL pipeline orchestrator for supply chain data processing
    
    Handles ingestion from multiple sources, data quality checks,
    transformation logic, and loading into PostgreSQL data warehouse.
    """
    
    def __init__(self, db_engine: Optional[Engine] = None):
        self.db_engine = db_engine or create_engine(DATABASE_URL, pool_pre_ping=True)
        self.connector = DataSourceConnector()
        self.metrics = ETLMetrics()
    
    def run_full_pipeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ETLMetrics:
        """
        Execute complete ETL pipeline for all data sources
        
        Args:
            start_date: Start date for data extraction (default: 30 days ago)
            end_date: End date for data extraction (default: now)
            
        Returns:
            ETLMetrics object with execution statistics
        """
        start_time = datetime.now()
        self.metrics = ETLMetrics()
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Starting ETL pipeline for period {start_date} to {end_date}")
        
        try:
            # Extract and load shipment data
            self._process_shipments(start_date, end_date)
            
            # Extract and load inventory data
            self._process_inventory(start_date, end_date)
            
            # Extract and load supplier performance data
            self._process_supplier_performance(start_date, end_date)
            
            # Extract and load geopolitical events
            self._process_geopolitical_events(start_date, end_date)
            
            # Clean up old data based on retention policy
            self._cleanup_old_data()
            
            # Update materialized views for analytics
            self._refresh_materialized_views()
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}", exc_info=True)
            self.metrics.errors.append(str(e))
        
        finally:
            self.metrics.execution_time_seconds = (
                datetime.now() - start_time
            ).total_seconds()
            logger.info(
                f"ETL pipeline completed. "
                f"Extracted: {self.metrics.records_extracted}, "
                f"Loaded: {self.metrics.records_loaded}, "
                f"Failed: {self.metrics.records_failed}"
            )
        
        return self.metrics
    
    def _process_shipments(self, start_date: datetime, end_date: datetime) -> None:
        """Extract, transform, and load shipment data"""
        logger.info("Processing shipment data")
        
        # Extract from multiple carrier APIs
        raw_data = []
        for carrier_endpoint in API_ENDPOINTS.get('carriers', []):
            data = self.connector.fetch_data(
                carrier_endpoint,
                params={
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            )
            if data:
                raw_data.extend(data.get('shipments', []))
        
        self.metrics.records_extracted += len(raw_data)
        
        if not raw_data:
            logger.warning("No shipment data extracted")
            return
        
        # Transform data
        df = pd.DataFrame(raw_data)
        df_transformed = self._transform_shipments(df)
        self.metrics.records_transformed += len(df_transformed)
        
        # Load to database
        loaded = self._load_to_database(
            df_transformed,
            table_name='shipments',
            conflict_columns=['shipment_id', 'tracking_number']
        )
        self.metrics.records_loaded += loaded
        self.metrics.records_failed += len(df_transformed) - loaded
    
    def _transform_shipments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to shipment data
        
        - Normalize location data
        - Calculate transit metrics
        - Enrich with carrier performance history
        - Detect anomalies in transit times
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Validate required fields
        required_fields = ['shipment_id', 'origin', 'destination', 'ship_date']
        if not all(field in df.columns for field in required_fields):
            logger.error(f"Missing required fields in shipment data")
            return pd.DataFrame()
        
        # Normalize location data (standardize country codes, city names)
        df['origin_normalized'] = df['origin'].apply(normalize_location_data)
        df['destination_normalized'] = df['destination'].apply(normalize_location_data)
        
        # Parse and standardize dates
        df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
        df['expected_delivery_date'] = pd.to_datetime(
            df.get('expected_delivery_date'),
            errors='coerce'
        )
        df['actual_delivery_date'] = pd.to_datetime(
            df.get('actual_delivery_date'),
            errors='coerce'
        )
        
        # Calculate transit metrics
        df['planned_transit_days'] = (
            df['expected_delivery_date'] - df['ship_date']
        ).dt.days
        
        df['actual_transit_days'] = (
            df['actual_delivery_date'] - df['ship_date']
        ).dt.days
        
        df['delay_days'] = df['actual_transit_days'] - df['planned_transit_days']
        df['is_delayed'] = df['delay_days'] > 0
        
        # Calculate route hash for grouping similar routes
        df['route_hash'] = df.apply(
            lambda row: hashlib.md5(
                f"{row['origin_normalized']}_{row['destination_normalized']}".encode()
            ).hexdigest()[:16],
            axis=1
        )
        
        # Add processing metadata
        df['etl_processed_at'] = datetime.now()
        df['data_quality_score'] = df.apply(self._calculate_data_quality_score, axis=1)
        
        # Filter out low quality records
        df = df[df['data_quality_score'] >= 0.6]
        
        return df
    
    def _process_inventory(self, start_date: datetime, end_date: datetime) -> None:
        """Extract, transform, and load inventory level data"""
        logger.info("Processing inventory data")
        
        raw_data = self.connector.fetch_data(
            API_ENDPOINTS.get('inventory_system'),
            params={
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
        )
        
        if not raw_data:
            logger.warning("No inventory data extracted")
            return
        
        self.metrics.records_extracted += len(raw_data.get('inventory_snapshots', []))
        
        df = pd.DataFrame(raw_data.get('inventory_snapshots', []))
        df_transformed = self._transform_inventory(df)
        self.metrics.records_transformed += len(df_transformed)
        
        loaded = self._load_to_database(
            df_transformed,
            table_name='inventory_levels',
            conflict_columns=['sku', 'warehouse_id', 'snapshot_date']
        )
        self.metrics.records_loaded += loaded
        self.metrics.records_failed += len(df_transformed) - loaded
    
    def _transform_inventory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to inventory data"""
        df = df.copy()
        
        # Validate data structure
        if not validate_inventory_data(df):
            logger.error("Inventory data validation failed")
            return pd.DataFrame()
        
        # Parse dates
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], errors='coerce')
        
        # Calculate inventory metrics
        df['inventory_turnover_ratio'] = df['units_sold'] / df['units_on_hand'].replace(0, np.nan)
        df['days_of_inventory'] = df['units_on_hand'] / (
            df['units_sold'] / 30
        ).replace(0, np.nan)
        
        # Flag stockout risks
        df['stockout_risk'] = df['days_of_inventory'] < 7
        
        # Calculate safety stock levels using historical demand volatility
        df['safety_stock_level'] = df.groupby('sku')['units_sold'].transform(
            lambda x: x.std() * 1.65  # 95% service level
        )
        
        # Identify slow-moving inventory
        df['is_slow_moving'] = df['inventory_turnover_ratio'] < 2
        
        # Add metadata
        df