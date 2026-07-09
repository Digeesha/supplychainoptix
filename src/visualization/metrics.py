"""
KPI calculation logic for supply chain metrics.

This module provides functions to calculate key performance indicators
used throughout the SupplyChainOptix dashboard, including inventory turnover,
fill rates, perfect order rate, and supply chain disruption metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MetricsCalculationError(Exception):
    """Raised when metrics calculation fails."""
    pass


def calculate_inventory_turnover(
    cost_of_goods_sold: float,
    avg_inventory_value: float,
    period_days: int = 365
) -> Dict[str, float]:
    """
    Calculate inventory turnover ratio and days of inventory.
    
    Args:
        cost_of_goods_sold: Total COGS for the period
        avg_inventory_value: Average inventory value
        period_days: Number of days in the period (default: 365)
    
    Returns:
        Dictionary containing turnover_ratio and days_of_inventory
    
    Raises:
        MetricsCalculationError: If inputs are invalid
    """
    try:
        if avg_inventory_value <= 0:
            logger.warning("Average inventory value must be positive")
            return {"turnover_ratio": 0.0, "days_of_inventory": 0.0}
        
        turnover_ratio = cost_of_goods_sold / avg_inventory_value
        days_of_inventory = period_days / turnover_ratio if turnover_ratio > 0 else 0.0
        
        return {
            "turnover_ratio": round(turnover_ratio, 2),
            "days_of_inventory": round(days_of_inventory, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating inventory turnover: {str(e)}")
        raise MetricsCalculationError(f"Failed to calculate inventory turnover: {str(e)}")


def calculate_fill_rate(
    orders_fulfilled: int,
    total_orders: int,
    by_units: bool = False,
    units_fulfilled: Optional[int] = None,
    total_units: Optional[int] = None
) -> float:
    """
    Calculate order fill rate or unit fill rate.
    
    Args:
        orders_fulfilled: Number of completely fulfilled orders
        total_orders: Total number of orders
        by_units: If True, calculate by units instead of orders
        units_fulfilled: Number of units fulfilled (required if by_units=True)
        total_units: Total units ordered (required if by_units=True)
    
    Returns:
        Fill rate as a percentage (0-100)
    """
    try:
        if by_units:
            if units_fulfilled is None or total_units is None:
                raise ValueError("units_fulfilled and total_units required when by_units=True")
            if total_units == 0:
                return 0.0
            return round((units_fulfilled / total_units) * 100, 2)
        else:
            if total_orders == 0:
                return 0.0
            return round((orders_fulfilled / total_orders) * 100, 2)
    except Exception as e:
        logger.error(f"Error calculating fill rate: {str(e)}")
        return 0.0


def calculate_perfect_order_rate(
    orders_df: pd.DataFrame,
    on_time_threshold_days: int = 0
) -> Dict[str, Union[float, int]]:
    """
    Calculate perfect order rate (complete, on-time, damage-free, with correct docs).
    
    Args:
        orders_df: DataFrame with columns: order_id, delivered_on_time, 
                   is_complete, is_damage_free, docs_correct
        on_time_threshold_days: Days of grace period for on-time delivery
    
    Returns:
        Dictionary with perfect_order_rate and count of perfect orders
    """
    try:
        if orders_df.empty:
            return {"perfect_order_rate": 0.0, "perfect_orders_count": 0, "total_orders": 0}
        
        required_cols = ["delivered_on_time", "is_complete", "is_damage_free", "docs_correct"]
        missing_cols = [col for col in required_cols if col not in orders_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Perfect order must satisfy all conditions
        perfect_orders = orders_df[
            (orders_df["delivered_on_time"] == True) &
            (orders_df["is_complete"] == True) &
            (orders_df["is_damage_free"] == True) &
            (orders_df["docs_correct"] == True)
        ]
        
        perfect_count = len(perfect_orders)
        total_count = len(orders_df)
        perfect_rate = (perfect_count / total_count * 100) if total_count > 0 else 0.0
        
        return {
            "perfect_order_rate": round(perfect_rate, 2),
            "perfect_orders_count": perfect_count,
            "total_orders": total_count
        }
    except Exception as e:
        logger.error(f"Error calculating perfect order rate: {str(e)}")
        raise MetricsCalculationError(f"Failed to calculate perfect order rate: {str(e)}")


def calculate_otif(
    deliveries_df: pd.DataFrame,
    scheduled_date_col: str = "scheduled_delivery_date",
    actual_date_col: str = "actual_delivery_date",
    quantity_ordered_col: str = "quantity_ordered",
    quantity_delivered_col: str = "quantity_delivered"
) -> Dict[str, float]:
    """
    Calculate On-Time In-Full (OTIF) delivery performance.
    
    Args:
        deliveries_df: DataFrame containing delivery records
        scheduled_date_col: Column name for scheduled delivery date
        actual_date_col: Column name for actual delivery date
        quantity_ordered_col: Column name for ordered quantity
        quantity_delivered_col: Column name for delivered quantity
    
    Returns:
        Dictionary with on_time_rate, in_full_rate, and otif_rate
    """
    try:
        if deliveries_df.empty:
            return {"on_time_rate": 0.0, "in_full_rate": 0.0, "otif_rate": 0.0}
        
        df = deliveries_df.copy()
        
        # Convert dates to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[scheduled_date_col]):
            df[scheduled_date_col] = pd.to_datetime(df[scheduled_date_col])
        if not pd.api.types.is_datetime64_any_dtype(df[actual_date_col]):
            df[actual_date_col] = pd.to_datetime(df[actual_date_col])
        
        # Calculate on-time deliveries
        df["is_on_time"] = df[actual_date_col] <= df[scheduled_date_col]
        on_time_rate = (df["is_on_time"].sum() / len(df)) * 100
        
        # Calculate in-full deliveries
        df["is_in_full"] = df[quantity_delivered_col] >= df[quantity_ordered_col]
        in_full_rate = (df["is_in_full"].sum() / len(df)) * 100
        
        # Calculate OTIF (both conditions must be met)
        df["is_otif"] = df["is_on_time"] & df["is_in_full"]
        otif_rate = (df["is_otif"].sum() / len(df)) * 100
        
        return {
            "on_time_rate": round(on_time_rate, 2),
            "in_full_rate": round(in_full_rate, 2),
            "otif_rate": round(otif_rate, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating OTIF: {str(e)}")
        raise MetricsCalculationError(f"Failed to calculate OTIF: {str(e)}")


def calculate_supply_chain_velocity(
    cycle_times_days: List[float]
) -> Dict[str, float]:
    """
    Calculate supply chain velocity metrics based on order cycle times.
    
    Args:
        cycle_times_days: List of cycle times in days
    
    Returns:
        Dictionary with mean, median, std, and velocity_score
    """
    try:
        if not cycle_times_days:
            return {
                "mean_cycle_time": 0.0,
                "median_cycle_time": 0.0,
                "std_cycle_time": 0.0,
                "velocity_score": 0.0
            }
        
        cycle_array = np.array(cycle_times_days)
        mean_time = float(np.mean(cycle_array))
        median_time = float(np.median(cycle_array))
        std_time = float(np.std(cycle_array))
        
        # Velocity score: inverse of cycle time normalized (higher is better)
        # Using median to reduce impact of outliers
        velocity_score = 100 / median_time if median_time > 0 else 0.0
        
        return {
            "mean_cycle_time": round(mean_time, 2),
            "median_cycle_time": round(median_time, 2),
            "std_cycle_time": round(std_time, 2),
            "velocity_score": round(velocity_score, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating supply chain velocity: {str(e)}")
        return {
            "mean_cycle_time": 0.0,
            "median_cycle_time": 0.0,
            "std_cycle_time": 0.0,
            "velocity_score": 0.0
        }


def calculate_disruption_impact_score(
    disruptions_df: pd.DataFrame,
    severity_col: str = "severity",
    affected_orders_col: str = "affected_orders",
    delay_hours_col: str = "delay_hours"
) -> Dict[str, Union[float, int]]:
    """
    Calculate aggregated disruption impact score across multiple disruptions.
    
    Args:
        disruptions_df: DataFrame containing disruption records
        severity_col: Column with severity ratings (1-10)
        affected_orders_col: Column with number of affected orders
        delay_hours_col: Column with delay duration in hours
    
    Returns:
        Dictionary with total_impact_score, avg_severity, total_affected_orders
    """
    try:
        if disruptions_df.empty:
            return {
                "total_impact_score": 0.0,
                "avg_severity": 0.0,
                "total_affected_orders": 0,
                "avg_delay_hours": 0.0
            }
        
        df = disruptions_df.copy()
        
        # Calculate weighted impact score
        # Impact = severity * affected_orders * (delay_hours / 24)
        df["impact"] = (
            df[severity_col] * 
            df[affected_orders_col] * 
            (df[delay_hours_col] / 24)
        )
        
        total_impact = df["impact"].sum()
        avg_severity = df[severity_col].mean()
        total_affected = df[affected_orders_col].sum()
        avg_delay = df[delay_hours_col].mean()
        
        return {
            "total_impact_score": round(total_impact, 2),
            "avg_severity": round(avg_severity, 2),
            "total_affected_orders": int(total_affected),
            "avg_delay_hours": round(avg_delay, 2)