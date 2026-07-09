"""
Tests for ETL pipeline components.

This module contains comprehensive tests for data extraction, transformation,
and loading operations in the supply chain data pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List
import json

from src.etl.extractor import DataExtractor
from src.etl.transformer import DataTransformer
from src.etl.loader import DataLoader
from src.etl.pipeline import ETLPipeline
from src.models.schemas import ShipmentRecord, InventoryRecord, GeopoliticalEvent


class TestDataExtractor:
    """Test suite for data extraction components."""

    @pytest.fixture
    def extractor(self):
        """Initialize extractor with mock database connection."""
        with patch('src.etl.extractor.psycopg2.connect') as mock_conn:
            extractor = DataExtractor(
                db_host='localhost',
                db_port=5432,
                db_name='test_db',
                db_user='test_user',
                db_password='test_pass'
            )
            return extractor

    def test_extract_shipment_data_success(self, extractor):
        """Test successful extraction of shipment data."""
        mock_data = pd.DataFrame({
            'shipment_id': ['SH001', 'SH002', 'SH003'],
            'origin': ['Shanghai', 'Hamburg', 'Los Angeles'],
            'destination': ['New York', 'Chicago', 'Seattle'],
            'departure_date': pd.date_range('2024-01-01', periods=3),
            'estimated_arrival': pd.date_range('2024-01-15', periods=3),
            'status': ['in_transit', 'delayed', 'delivered'],
            'carrier': ['MSC', 'Maersk', 'COSCO']
        })

        with patch.object(extractor, '_execute_query', return_value=mock_data):
            result = extractor.extract_shipment_data(days_back=30)
            
            assert len(result) == 3
            assert 'shipment_id' in result.columns
            assert result['status'].iloc[0] == 'in_transit'

    def test_extract_inventory_data_with_date_filter(self, extractor):
        """Test inventory extraction with date filtering."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        mock_data = pd.DataFrame({
            'sku': ['SKU001', 'SKU002', 'SKU003'],
            'warehouse_id': ['WH01', 'WH01', 'WH02'],
            'quantity': [150, 200, 75],
            'reorder_point': [100, 150, 50],
            'last_updated': pd.date_range('2024-01-15', periods=3)
        })

        with patch.object(extractor, '_execute_query', return_value=mock_data):
            result = extractor.extract_inventory_data(start_date, end_date)
            
            assert len(result) == 3
            assert all(result['quantity'] >= 0)

    def test_extract_geopolitical_events(self, extractor):
        """Test extraction of geopolitical events from external API."""
        mock_events = [
            {
                'event_id': 'EV001',
                'event_type': 'port_strike',
                'location': 'Shanghai',
                'severity': 'high',
                'start_date': '2024-01-10',
                'end_date': '2024-01-15',
                'impact_radius_km': 500
            },
            {
                'event_id': 'EV002',
                'event_type': 'weather_disruption',
                'location': 'Hamburg',
                'severity': 'medium',
                'start_date': '2024-01-12',
                'end_date': None,
                'impact_radius_km': 200
            }
        ]

        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'events': mock_events}
            
            result = extractor.extract_geopolitical_events(days_ahead=14)
            
            assert len(result) == 2
            assert result['event_type'].iloc[0] == 'port_strike'

    def test_extract_with_connection_failure(self, extractor):
        """Test handling of database connection failures."""
        with patch.object(extractor, '_execute_query', side_effect=ConnectionError("DB unavailable")):
            with pytest.raises(ConnectionError):
                extractor.extract_shipment_data(days_back=7)

    def test_extract_empty_result(self, extractor):
        """Test extraction when no data is available."""
        empty_df = pd.DataFrame()
        
        with patch.object(extractor, '_execute_query', return_value=empty_df):
            result = extractor.extract_shipment_data(days_back=30)
            
            assert len(result) == 0
            assert isinstance(result, pd.DataFrame)


class TestDataTransformer:
    """Test suite for data transformation operations."""

    @pytest.fixture
    def transformer(self):
        """Initialize transformer instance."""
        return DataTransformer()

    @pytest.fixture
    def sample_shipment_data(self):
        """Generate sample shipment data for testing."""
        return pd.DataFrame({
            'shipment_id': ['SH001', 'SH002', 'SH003', 'SH004'],
            'origin': ['Shanghai', 'Hamburg', 'Los Angeles', 'Shanghai'],
            'destination': ['New York', 'Chicago', 'Seattle', 'New York'],
            'departure_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
            'estimated_arrival': pd.to_datetime(['2024-01-15', '2024-01-18', '2024-01-10', '2024-01-16']),
            'actual_arrival': pd.to_datetime(['2024-01-17', '2024-01-18', '2024-01-11', None]),
            'status': ['delivered', 'delivered', 'delivered', 'in_transit'],
            'carrier': ['MSC', 'Maersk', 'COSCO', 'MSC'],
            'value_usd': [50000, 75000, 30000, 60000]
        })

    def test_calculate_transit_times(self, transformer, sample_shipment_data):
        """Test calculation of actual and expected transit times."""
        result = transformer.calculate_transit_times(sample_shipment_data)
        
        assert 'expected_transit_days' in result.columns
        assert 'actual_transit_days' in result.columns
        assert 'delay_days' in result.columns
        
        # Verify specific calculations
        assert result.loc[0, 'expected_transit_days'] == 14
        assert result.loc[0, 'actual_transit_days'] == 16
        assert result.loc[0, 'delay_days'] == 2

    def test_normalize_locations(self, transformer, sample_shipment_data):
        """Test location standardization and geocoding."""
        result = transformer.normalize_locations(sample_shipment_data)
        
        assert 'origin_normalized' in result.columns
        assert 'destination_normalized' in result.columns
        assert 'origin_lat' in result.columns
        assert 'origin_lon' in result.columns

    def test_calculate_route_metrics(self, transformer, sample_shipment_data):
        """Test calculation of route-specific metrics."""
        # First normalize locations to get coordinates
        data_with_coords = transformer.normalize_locations(sample_shipment_data)
        result = transformer.calculate_route_metrics(data_with_coords)
        
        assert 'route_id' in result.columns
        assert 'distance_km' in result.columns
        assert 'route_frequency' in result.columns

    def test_detect_anomalies(self, transformer, sample_shipment_data):
        """Test anomaly detection in shipment data."""
        data_with_transit = transformer.calculate_transit_times(sample_shipment_data)
        result = transformer.detect_anomalies(data_with_transit)
        
        assert 'is_anomaly' in result.columns
        assert 'anomaly_score' in result.columns
        assert result['is_anomaly'].dtype == bool

    def test_aggregate_carrier_performance(self, transformer, sample_shipment_data):
        """Test aggregation of carrier performance metrics."""
        data_with_transit = transformer.calculate_transit_times(sample_shipment_data)
        result = transformer.aggregate_carrier_performance(data_with_transit)
        
        assert 'carrier' in result.columns
        assert 'avg_delay_days' in result.columns
        assert 'on_time_percentage' in result.columns
        assert 'total_shipments' in result.columns
        
        # MSC should have 2 shipments
        msc_data = result[result['carrier'] == 'MSC']
        assert len(msc_data) == 1
        assert msc_data.iloc[0]['total_shipments'] == 2

    def test_transform_inventory_data(self, transformer):
        """Test inventory data transformation including safety stock calculations."""
        inventory_data = pd.DataFrame({
            'sku': ['SKU001', 'SKU002', 'SKU003'],
            'warehouse_id': ['WH01', 'WH01', 'WH02'],
            'quantity': [150, 50, 200],
            'reorder_point': [100, 80, 150],
            'daily_demand_avg': [10, 15, 8],
            'daily_demand_std': [2, 5, 1.5],
            'lead_time_days': [14, 21, 10]
        })
        
        result = transformer.transform_inventory_data(inventory_data)
        
        assert 'days_until_stockout' in result.columns
        assert 'safety_stock' in result.columns
        assert 'reorder_needed' in result.columns
        assert result['reorder_needed'].dtype == bool

    def test_enrich_with_geopolitical_data(self, transformer, sample_shipment_data):
        """Test enrichment of shipment data with geopolitical events."""
        geo_events = pd.DataFrame({
            'event_id': ['EV001', 'EV002'],
            'event_type': ['port_strike', 'weather'],
            'location': ['Shanghai', 'New York'],
            'severity': ['high', 'medium'],
            'start_date': pd.to_datetime(['2024-01-01', '2024-01-15']),
            'end_date': pd.to_datetime(['2024-01-05', '2024-01-16']),
            'lat': [31.2304, 40.7128],
            'lon': [121.4737, -74.0060]
        })
        
        data_with_coords = transformer.normalize_locations(sample_shipment_data)
        result = transformer.enrich_with_geopolitical_data(data_with_coords, geo_events)
        
        assert 'affected_by_events' in result.columns
        assert 'event_risk_score' in result.columns

    def test_handle_missing_values(self, transformer):
        """Test handling of missing values in transformation."""
        data_with_nulls = pd.DataFrame({
            'shipment_id': ['SH001', 'SH002', 'SH003'],
            'origin