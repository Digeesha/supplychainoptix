"""
SupplyChainOptix - Real-time Supply Chain Disruption Prediction Dashboard

This is the main entry point for the Streamlit application that provides
supply chain disruption predictions, anomaly detection, and inventory recommendations.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules
try:
    from src.database.db_manager import DatabaseManager
    from src.ml.prophet_forecaster import ProphetForecaster
    from src.ml.anomaly_detector import IsolationForestDetector
    from src.data.data_processor import DataProcessor
    from src.external.geopolitical_tracker import GeopoliticalTracker
    from src.analytics.inventory_optimizer import InventoryOptimizer
    from src.utils.cache_manager import CacheManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    st.error("Application initialization failed. Please check the logs.")
    sys.exit(1)

# Page configuration
st.set_page_config(
    page_title="SupplyChainOptix",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .critical-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_services() -> Dict:
    """
    Initialize all backend services and connections.
    
    Returns:
        Dictionary containing initialized service instances
    """
    try:
        db_manager = DatabaseManager(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "supplychainoptix"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
        
        cache_manager = CacheManager(ttl=int(os.getenv("CACHE_TTL", "300")))
        data_processor = DataProcessor()
        prophet_forecaster = ProphetForecaster()
        anomaly_detector = IsolationForestDetector(contamination=0.1)
        geopolitical_tracker = GeopoliticalTracker(api_key=os.getenv("GEO_API_KEY"))
        inventory_optimizer = InventoryOptimizer()
        
        logger.info("All services initialized successfully")
        
        return {
            "db": db_manager,
            "cache": cache_manager,
            "processor": data_processor,
            "forecaster": prophet_forecaster,
            "detector": anomaly_detector,
            "geo_tracker": geopolitical_tracker,
            "optimizer": inventory_optimizer
        }
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        st.error(f"Failed to initialize services: {str(e)}")
        return None


def render_header():
    """Render application header with branding and status indicators."""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown('<div class="main-header">📦 SupplyChainOptix</div>', unsafe_allow_html=True)
        st.caption("Real-time Supply Chain Disruption Prediction & Analytics")
    
    with col2:
        st.metric("System Status", "🟢 Active", delta="All systems operational")
    
    with col3:
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.metric("Last Update", last_update, delta="2 min ago")


def render_sidebar(services: Dict) -> Dict:
    """
    Render sidebar with filters and configuration options.
    
    Args:
        services: Dictionary of initialized services
        
    Returns:
        Dictionary containing selected filter values
    """
    st.sidebar.title("⚙️ Configuration")
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    date_range = st.sidebar.date_input(
        "Select Period",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )
    
    # Supplier selection
    st.sidebar.subheader("Suppliers")
    try:
        suppliers = services["db"].get_active_suppliers()
        selected_suppliers = st.sidebar.multiselect(
            "Filter by Suppliers",
            options=suppliers,
            default=suppliers[:5] if len(suppliers) > 5 else suppliers
        )
    except Exception as e:
        logger.error(f"Failed to fetch suppliers: {e}")
        selected_suppliers = []
        st.sidebar.error("Unable to load suppliers")
    
    # Region selection
    st.sidebar.subheader("Regions")
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East", "Africa"]
    selected_regions = st.sidebar.multiselect(
        "Filter by Regions",
        options=regions,
        default=["North America", "Europe"]
    )
    
    # Risk threshold
    st.sidebar.subheader("Risk Settings")
    risk_threshold = st.sidebar.slider(
        "Risk Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Set minimum risk score for alerts"
    )
    
    # Forecast horizon
    forecast_days = st.sidebar.selectbox(
        "Forecast Horizon (days)",
        options=[7, 14, 21, 30],
        index=1
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    return {
        "date_range": date_range,
        "suppliers": selected_suppliers,
        "regions": selected_regions,
        "risk_threshold": risk_threshold,
        "forecast_days": forecast_days
    }


@st.cache_data(ttl=300)
def load_supply_chain_data(_services: Dict, filters: Dict) -> pd.DataFrame:
    """
    Load and process supply chain data based on filters.
    
    Args:
        _services: Dictionary of initialized services
        filters: Filter parameters from sidebar
        
    Returns:
        Processed DataFrame with supply chain metrics
    """
    try:
        # Fetch raw data from database
        raw_data = _services["db"].fetch_logistics_data(
            start_date=filters["date_range"][0],
            end_date=filters["date_range"][1],
            suppliers=filters["suppliers"],
            regions=filters["regions"]
        )
        
        # Process and clean data
        processed_data = _services["processor"].clean_and_transform(raw_data)
        
        logger.info(f"Loaded {len(processed_data)} records")
        return processed_data
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()


def render_kpi_metrics(data: pd.DataFrame, filters: Dict):
    """
    Render key performance indicator metrics at the top of dashboard.
    
    Args:
        data: Processed supply chain data
        filters: Current filter settings
    """
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Calculate metrics
        avg_delay = data['delay_hours'].mean() if 'delay_hours' in data.columns else 0
        disruption_count = data[data['is_disrupted'] == True].shape[0] if 'is_disrupted' in data.columns else 0
        at_risk_shipments = data[data['risk_score'] >= filters['risk_threshold']].shape[0] if 'risk_score' in data.columns else 0
        total_shipments = len(data)
        
        # Display metrics
        with col1:
            delta_delay = f"{((avg_delay - 12) / 12 * 100):.1f}%" if avg_delay > 0 else "0%"
            st.metric(
                "Avg Delay (hours)",
                f"{avg_delay:.1f}",
                delta=delta_delay,
                delta_color="inverse"
            )
        
        with col2:
            disruption_rate = (disruption_count / total_shipments * 100) if total_shipments > 0 else 0
            st.metric(
                "Active Disruptions",
                disruption_count,
                delta=f"{disruption_rate:.1f}%",
                delta_color="inverse"
            )
        
        with col3:
            risk_rate = (at_risk_shipments / total_shipments * 100) if total_shipments > 0 else 0
            st.metric(
                "At-Risk Shipments",
                at_risk_shipments,
                delta=f"{risk_rate:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Total Shipments",
                total_shipments,
                delta="Last 30 days"
            )
            
    except Exception as e:
        logger.error(f"Error calculating KPIs: {e}")
        st.error("Unable to calculate KPI metrics")


def render_disruption_forecast(services: Dict, data: pd.DataFrame, forecast_days: int):
    """
    Render disruption forecast visualization using Prophet model.
    
    Args:
        services: Dictionary of initialized services
        data: Historical supply chain data
        forecast_days: Number of days to forecast ahead
    """
    st.subheader("📈 Disruption Forecast (7-14 Days Ahead)")
    
    try:
        # Prepare time series data
        if 'date' not in data.columns or 'disruption_score' not in data.columns:
            st.warning("Insufficient data for forecasting")
            return
        
        ts_data = data[['date', 'disruption_score']].copy()
        ts_data.columns = ['ds', 'y']
        
        # Generate forecast
        with st.spinner("Generating forecast..."):
            forecast_df = services["forecaster"].predict(ts_data, periods=forecast_days)
        
        # Create visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(