from typing import Dict, Any, List, Optional
from langchain_core.tools import tool


@tool()
def generate_tabular_data(
    description: str, 
    schema: Optional[Dict[str, Any]] = None,
    size: int = 100,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates synthetic tabular data based on a schema and description.
    
    Args:
        description: Natural language description of the data to generate
        schema: Optional schema definition for the data structure
        size: Number of data records to generate (default: 100)
        metadata: Additional metadata to guide generation
        
    Returns:
        Generated synthetic data matching the requested specification
    """
    pass


@tool()
def generate_timeseries_data(
    description: str,
    start_date: str,
    end_date: str,
    frequency: str = "daily",
    fields: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    size: int = 100
) -> Dict[str, Any]:
    """
    Generates synthetic time series data based on description and parameters.
    
    Args:
        description: Natural language description of the time series data
        start_date: Start date for the time series (YYYY-MM-DD)
        end_date: End date for the time series (YYYY-MM-DD)
        frequency: Data frequency (daily, hourly, minutely, etc.)
        fields: Optional list of field names to generate
        patterns: Optional patterns to include (trend, seasonality, etc.)
        size: Number of data points to generate
        
    Returns:
        Generated time series data
    """
    pass 