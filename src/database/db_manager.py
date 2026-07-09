```python
"""
PostgreSQL Database Manager for SupplyChainOptix

Handles all database connections, query execution, and connection pooling
for supply chain data operations.
"""

import os
import logging
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extensions import connection, cursor
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass


class QueryExecutionError(Exception):
    """Raised when query execution fails"""
    pass


class DBManager:
    """
    PostgreSQL database manager with connection pooling and error handling.
    
    Provides methods for executing queries, managing transactions, and
    retrieving supply chain data with proper resource cleanup.
    """
    
    _instance: Optional['DBManager'] = None
    _connection_pool: Optional[pool.ThreadedConnectionPool] = None
    
    def __new__(cls):
        """Singleton pattern to ensure single connection pool"""
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize database manager with connection pool"""
        if self._connection_pool is None:
            self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """
        Create connection pool using environment variables.
        
        Raises:
            DatabaseConnectionError: If connection pool creation fails
        """
        try:
            self._connection_pool = pool.ThreadedConnectionPool(
                minconn=int(os.getenv('DB_MIN_CONNECTIONS', '2')),
                maxconn=int(os.getenv('DB_MAX_CONNECTIONS', '10')),
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', '5432')),
                database=os.getenv('DB_NAME', 'supplychain_optix'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', ''),
                connect_timeout=10,
                options='-c statement_timeout=30000'  # 30 second query timeout
            )
            logger.info("Database connection pool initialized successfully")
        except psycopg2.Error as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise DatabaseConnectionError(f"Connection pool initialization failed: {e}")
    
    @contextmanager
    def get_connection(self) -> connection:
        """
        Context manager for database connections.
        
        Yields:
            connection: PostgreSQL connection from pool
            
        Raises:
            DatabaseConnectionError: If connection cannot be obtained
        """
        conn = None
        try:
            conn = self._connection_pool.getconn()
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseConnectionError(f"Failed to get connection: {e}")
        finally:
            if conn:
                self._connection_pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = False) -> cursor:
        """
        Context manager for database cursors with automatic cleanup.
        
        Args:
            dict_cursor: If True, returns DictCursor for named column access
            
        Yields:
            cursor: PostgreSQL cursor
        """
        with self.get_connection() as conn:
            cur = None
            try:
                if dict_cursor:
                    cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                else:
                    cur = conn.cursor()
                yield cur
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Cursor operation failed: {e}")
                raise
            finally:
                if cur:
                    cur.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(psycopg2.OperationalError)
    )
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = True
    ) -> Optional[List[Tuple]]:
        """
        Execute a SQL query with retry logic.
        
        Args:
            query: SQL query string
            params: Query parameters for safe parameterization
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, None otherwise
            
        Raises:
            QueryExecutionError: If query execution fails after retries
        """
        try:
            with self.get_cursor() as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                return None
        except psycopg2.Error as e:
            logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
            raise QueryExecutionError(f"Failed to execute query: {e}")
    
    def execute_batch(
        self,
        query: str,
        data: List[Tuple],
        page_size: int = 1000
    ) -> None:
        """
        Execute batch insert/update operations efficiently.
        
        Args:
            query: SQL query with placeholders
            data: List of tuples containing parameter values
            page_size: Number of records per batch
            
        Raises:
            QueryExecutionError: If batch execution fails
        """
        try:
            with self.get_cursor() as cur:
                extras.execute_batch(cur, query, data, page_size=page_size)
            logger.info(f"Batch operation completed: {len(data)} records")
        except psycopg2.Error as e:
            logger.error(f"Batch execution failed: {e}")
            raise QueryExecutionError(f"Failed to execute batch: {e}")
    
    def fetch_dataframe(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame containing query results
            
        Raises:
            QueryExecutionError: If query execution fails
        """
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"DataFrame fetch failed: {e}")
            raise QueryExecutionError(f"Failed to fetch DataFrame: {e}")
    
    def get_shipment_data(
        self,
        start_date: datetime,
        end_date: datetime,
        supplier_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Retrieve shipment data for specified date range and suppliers.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            supplier_ids: Optional list of supplier IDs to filter
            
        Returns:
            DataFrame with shipment records
        """
        query = """
            SELECT 
                s.shipment_id,
                s.supplier_id,
                s.origin_location,
                s.destination_location,
                s.expected_delivery_date,
                s.actual_delivery_date,
                s.status,
                s.carrier,
                s.product_category,
                s.quantity,
                s.value_usd,
                s.lead_time_days,
                s.created_at,
                s.updated_at
            FROM shipments s
            WHERE s.created_at BETWEEN %s AND %s
        """
        
        params = [start_date, end_date]
        
        if supplier_ids:
            query += " AND s.supplier_id = ANY(%s)"
            params.append(supplier_ids)
        
        query += " ORDER BY s.created_at DESC"
        
        return self.fetch_dataframe(query, tuple(params))
    
    def get_inventory_snapshot(
        self,
        location_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Get current inventory levels across warehouses.
        
        Args:
            location_ids: Optional list of warehouse location IDs
            
        Returns:
            DataFrame with current inventory levels
        """
        query = """
            SELECT 
                i.inventory_id,
                i.location_id,
                l.location_name,
                l.country,
                i.product_id,
                p.product_name,
                p.category,
                i.current_stock,
                i.reorder_point,
                i.safety_stock,
                i.average_daily_demand,
                i.days_of_supply,
                i.last_restock_date,
                i.updated_at
            FROM inventory i
            JOIN locations l ON i.location_id = l.location_id
            JOIN products p ON i.product_id = p.product_id
            WHERE i.is_active = true
        """
        
        params = []
        
        if location_ids:
            query += " AND i.location_id = ANY(%s)"
            params.append(location_ids)
        
        query += " ORDER BY i.days_of_supply ASC"
        
        return self.fetch_dataframe(query, tuple(params) if params else None)
    
    def get_disruption_events(
        self,
        lookback_days: int = 90,
        event_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical disruption events for ML training.
        
        Args:
            lookback_days: Number of days to look back
            event_types: Optional filter for specific event types
            
        Returns:
            DataFrame with disruption events
        """
        start_date = datetime.now() - timedelta(days=lookback_days)
        
        query = """
            SELECT 
                de.event_id,
                de.event_type,
                de.severity,
                de.affected_region,
                de.start_date,
                de.end_date,
                de.description,
                de.impact_score,
                de.supplier_count_affected,
                de.shipment_count_affected,
                de.resolution_time_hours,
                de.created_at
            FROM disruption_events de
            WHERE de.start_date >= %s
        """
        
        params = [start_date]
        
        if event_types:
            query += " AND de.event_type = ANY(%s)"
            params.append(event_types)
        
        query += " ORDER BY de.start_date DESC"
        
        return self.fetch_dataframe(query, tuple(params))
    
    def get_supplier_performance_metrics(
        self,
        supplier_ids: Optional[List[int]] = None,
        days: int = 180
    ) -> pd.DataFrame:
        """
        Calculate supplier performance metrics for risk assessment.
        
        Args:
            supplier_ids: Optional list of supplier IDs
            days: Number of days for metric calculation
            
        Returns:
            DataFrame with supplier performance data
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
            WITH supplier_metrics AS (
                SELECT 
                    s.supplier_id,
                    s.supplier_name,
                    s.country,
                    s.region,
                    COUNT(sh.shipment_id) as total_shipments,
                    AVG(sh.