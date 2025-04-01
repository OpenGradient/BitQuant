from typing import Dict, Any, List, Optional, Union, Callable
from langchain_core.tools import tool
import json
import datetime
import random
import uuid
import re
from enum import Enum


class DataType(Enum):
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    DATE = "date"
    OBJECT = "object"
    ARRAY = "array"


class DataPattern(Enum):
    RANDOM = "random"
    SEQUENTIAL = "sequential"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    NORMAL = "normal"
    UNIFORM = "uniform"


class DataFormat(Enum):
    EMAIL = "email"
    URL = "url"
    DATE = "date"
    DATE_TIME = "date-time"
    TIME = "time"
    PHONE = "phone"
    IP = "ip"
    UUID = "uuid"
    CURRENCY = "currency"
    ZIP_CODE = "zip"
    ADDRESS = "address"
    PERSON_NAME = "name"
    CUSTOM = "custom"


@tool()
def generate_synthetic_data(
    rfd: Dict[str, Any],
    size: int = 100,
    format_type: str = "json",
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates synthetic data based on a Request for Data (RFD) schema.
    This is the main general-purpose tool for generating data from any RFD definition.
    
    Args:
        rfd: The Request for Data (RFD) object containing schema and metadata
        size: Number of records to generate (default: 100)
        format_type: Output format (json, csv, etc.)
        options: Additional options to customize generation
        
    Returns:
        Generated synthetic data matching the RFD schema
    """
    # Validate RFD structure
    validate_rfd(rfd)
    
    # Extract key information from RFD
    schema = rfd.get("schema", {})
    description = rfd.get("description", "")
    rfd_id = rfd.get("rfd_id", str(uuid.uuid4()))
    
    # Check if the RFD is for time series data
    is_timeseries = is_timeseries_schema(schema, description)
    
    # Generate data based on the schema
    if is_timeseries:
        dataset = generate_timeseries_dataset(schema, description, size, options)
    else:
        dataset = generate_tabular_dataset(schema, description, size, options)
    
    # Format the output
    result = {
        "data": dataset,
        "metadata": {
            "rfd_id": rfd_id,
            "name": rfd.get("name", "Synthetic Dataset"),
            "description": description,
            "size": len(dataset),
            "generated_at": datetime.datetime.now().isoformat(),
            "format": format_type
        }
    }
    
    return result


@tool()
def generate_tabular_data(
    rfd: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None, 
    schema: Optional[Dict[str, Any]] = None,
    size: int = 100,
    metadata: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates synthetic tabular data based on an RFD or schema.
    
    Args:
        rfd: Request for Data object with schema definition
        description: Natural language description of the data to generate
        schema: Optional schema definition if not using RFD
        size: Number of data records to generate (default: 100)
        metadata: Additional metadata to guide generation
        options: Additional options to customize generation
        
    Returns:
        Generated synthetic data matching the requested specification
    """
    if rfd:
        # Use the more general-purpose tool
        return generate_synthetic_data(rfd, size, options=options)
    
    # Create a minimal RFD structure if not provided
    if schema is None:
        schema = infer_schema_from_description(description or "")
    
    minimal_rfd = {
        "rfd_id": str(uuid.uuid4()),
        "name": "Generated Dataset",
        "description": description or "Tabular dataset",
        "schema": schema
    }
    
    if metadata:
        minimal_rfd["metadata"] = metadata
    
    return generate_synthetic_data(minimal_rfd, size, options=options)


@tool()
def generate_timeseries_data(
    rfd: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = "daily",
    fields: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    size: int = 100,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates synthetic time series data based on RFD or parameters.
    
    Args:
        rfd: Request for Data object with schema definition
        description: Natural language description of the time series data
        start_date: Start date for the time series (YYYY-MM-DD)
        end_date: End date for the time series (YYYY-MM-DD)
        frequency: Data frequency (daily, hourly, minutely, etc.)
        fields: Optional list of field names to generate
        patterns: Optional patterns to include (trend, seasonality, etc.)
        size: Number of data points to generate
        options: Additional options to customize generation
        
    Returns:
        Generated time series data
    """
    if rfd:
        # If RFD includes specific time series parameters, extract them
        ts_options = options or {}
        
        if start_date:
            ts_options["start_date"] = start_date
        if end_date:
            ts_options["end_date"] = end_date
        if frequency:
            ts_options["frequency"] = frequency
        if fields:
            ts_options["fields"] = fields
        if patterns:
            ts_options["patterns"] = patterns
            
        return generate_synthetic_data(rfd, size, options=ts_options)
    
    # Create a minimal RFD for time series if not provided
    schema = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "format": "date"
            }
        },
        "required": ["date"]
    }
    
    # Add the specified fields or a default value field
    if fields:
        for field in fields:
            schema["properties"][field] = {"type": "number"}
            schema["required"].append(field)
    else:
        schema["properties"]["value"] = {"type": "number"}
        schema["required"].append("value")
    
    minimal_rfd = {
        "rfd_id": str(uuid.uuid4()),
        "name": "Time Series Dataset",
        "description": description or "Time series dataset",
        "schema": schema,
        "metadata": {
            "time_series": True,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency,
            "patterns": patterns
        }
    }
    
    return generate_synthetic_data(minimal_rfd, size, options=options)


@tool()
def process_rfd(
    rfd: Dict[str, Any],
    size: int = 100,
    additional_constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Processes a Request for Data (RFD) object and generates data according to its schema.
    This is an alias for generate_synthetic_data for backward compatibility.
    
    Args:
        rfd: The Request for Data object
        size: Number of data records to generate
        additional_constraints: Optional constraints for data generation
        
    Returns:
        Generated data conforming to the RFD schema
    """
    return generate_synthetic_data(rfd, size, options=additional_constraints)


@tool()
def analyze_rfd(rfd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes a Request for Data (RFD) object and returns information about its structure.
    
    Args:
        rfd: The Request for Data object to analyze
        
    Returns:
        Analysis of the RFD structure including field types, constraints, and suggestions
    """
    # Validate and extract RFD components
    valid, validation_errors = validate_rfd(rfd, return_errors=True)
    
    # Extract schema information
    schema = rfd.get("schema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Count field types
    field_types = {}
    for field_name, field_def in properties.items():
        field_type = field_def.get("type", "unknown")
        if field_type not in field_types:
            field_types[field_type] = 0
        field_types[field_type] += 1
    
    # Determine if it's time series
    is_timeseries = is_timeseries_schema(schema, rfd.get("description", ""))
    
    # Check for constraints
    constraints = []
    for field_name, field_def in properties.items():
        if "minimum" in field_def or "maximum" in field_def:
            constraints.append(f"{field_name} has numeric constraints")
        if "minLength" in field_def or "maxLength" in field_def:
            constraints.append(f"{field_name} has length constraints")
        if "enum" in field_def:
            constraints.append(f"{field_name} has enumerated values")
        if "pattern" in field_def:
            constraints.append(f"{field_name} has a regex pattern")
    
    # Generate analysis
    analysis = {
        "rfd_id": rfd.get("rfd_id", "unknown"),
        "name": rfd.get("name", "Unnamed RFD"),
        "is_valid": valid,
        "validation_issues": validation_errors if not valid else [],
        "field_count": len(properties),
        "required_fields": required,
        "field_types": field_types,
        "is_timeseries": is_timeseries,
        "constraints": constraints,
        "suggestions": generate_suggestions(rfd)
    }
    
    return analysis


@tool()
def validate_rfd_tool(rfd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a Request for Data (RFD) object and returns validation results.
    
    Args:
        rfd: The Request for Data object to validate
        
    Returns:
        Validation results including any errors or warnings
    """
    valid, errors = validate_rfd(rfd, return_errors=True)
    
    return {
        "is_valid": valid,
        "errors": errors,
        "rfd_id": rfd.get("rfd_id", "unknown")
    }


# Helper functions

def validate_rfd(rfd: Dict[str, Any], return_errors: bool = False) -> Union[bool, tuple]:
    """Validates the structure of an RFD object."""
    errors = []
    
    # Check for required fields
    if not isinstance(rfd, dict):
        errors.append("RFD must be a dictionary")
        if return_errors:
            return False, errors
        return False
    
    # Check for required fields
    for field in ["schema"]:
        if field not in rfd:
            errors.append(f"Missing required field: {field}")
    
    # Validate schema if present
    if "schema" in rfd:
        schema = rfd["schema"]
        if not isinstance(schema, dict):
            errors.append("Schema must be a dictionary")
        else:
            # Check schema structure
            if "properties" not in schema:
                errors.append("Schema missing 'properties' field")
            elif not isinstance(schema["properties"], dict):
                errors.append("Schema properties must be a dictionary")
            
            # Check property definitions
            if "properties" in schema and isinstance(schema["properties"], dict):
                for prop_name, prop_def in schema["properties"].items():
                    if not isinstance(prop_def, dict):
                        errors.append(f"Property '{prop_name}' definition must be a dictionary")
                    elif "type" not in prop_def:
                        errors.append(f"Property '{prop_name}' missing 'type' field")
    
    if return_errors:
        return len(errors) == 0, errors
    
    return len(errors) == 0


def is_timeseries_schema(schema: Dict[str, Any], description: str) -> bool:
    """Detects if a schema represents time series data."""
    # Check properties for date/time fields
    properties = schema.get("properties", {})
    
    # Look for date/time fields
    has_date_field = any(
        field_name.lower() in ["date", "time", "timestamp", "datetime"] or
        (isinstance(field_def, dict) and field_def.get("format") in ["date", "date-time", "time"]) or
        (isinstance(field_def, dict) and field_def.get("type") == "date")
        for field_name, field_def in properties.items()
    )
    
    # Check description for time-related keywords
    time_keywords = ["time series", "temporal", "timeseries", "time-series", 
                     "sequential", "chronological", "historical"]
    has_time_desc = any(keyword in description.lower() for keyword in time_keywords)
    
    return has_date_field or has_time_desc


def generate_tabular_dataset(
    schema: Dict[str, Any], 
    description: str, 
    size: int, 
    options: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generates a tabular dataset based on the schema."""
    options = options or {}
    properties = schema.get("properties", {})
    dataset = []
    
    # Generate each record
    for i in range(size):
        record = {}
        for field_name, field_def in properties.items():
            field_type = field_def.get("type", "string")
            field_format = field_def.get("format")
            field_enum = field_def.get("enum")
            
            # Generate value based on type and format
            value = generate_field_value(field_type, field_format, field_enum, field_def, i, size, options)
            record[field_name] = value
            
        dataset.append(record)
    
    return dataset


def generate_timeseries_dataset(
    schema: Dict[str, Any], 
    description: str, 
    size: int, 
    options: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generates a time series dataset."""
    options = options or {}
    properties = schema.get("properties", {})
    
    # Determine date field and value fields
    date_field = next(
        (field for field, def_ in properties.items() 
         if field.lower() in ["date", "time", "timestamp"] or 
         def_.get("format") in ["date", "date-time"]),
        "date"  # Default field name if none found
    )
    
    # Extract time series parameters
    start_date = options.get("start_date", "2023-01-01")
    end_date = options.get("end_date")
    frequency = options.get("frequency", "daily")
    patterns = options.get("patterns", ["random"])
    
    # Generate date range
    dates = generate_date_sequence(start_date, end_date, frequency, size)
    
    # Generate data for each date
    dataset = []
    for i, date in enumerate(dates):
        record = {date_field: date}
        
        # Generate values for each field
        for field_name, field_def in properties.items():
            if field_name == date_field:
                continue
                
            field_type = field_def.get("type", "number")
            pattern = options.get(f"pattern_{field_name}", patterns[0] if patterns else "random")
            
            # Don't generate for the date field
            if field_name != date_field:
                if field_type in ["number", "integer"]:
                    # Generate time series value based on pattern
                    value = generate_timeseries_value(
                        field_type, pattern, i, size, 
                        field_def.get("minimum"), field_def.get("maximum")
                    )
                else:
                    # For non-numeric fields, use regular generation
                    value = generate_field_value(
                        field_type, field_def.get("format"), field_def.get("enum"), 
                        field_def, i, size, options
                    )
                    
                record[field_name] = value
        
        dataset.append(record)
    
    return dataset


def generate_field_value(
    field_type: str,
    field_format: Optional[str],
    field_enum: Optional[List[Any]],
    field_def: Dict[str, Any],
    index: int,
    total_size: int,
    options: Dict[str, Any]
) -> Any:
    """Generates a single field value based on its type and constraints."""
    # If enum is specified, select from the enum values
    if field_enum:
        return random.choice(field_enum)
    
    # Generate based on format first if specified
    if field_format:
        return generate_formatted_value(field_format, field_def)
    
    # Generate based on type
    if field_type == DataType.STRING.value:
        min_length = field_def.get("minLength", 1)
        max_length = field_def.get("maxLength", 20)
        return generate_random_string(min_length, max_length)
        
    elif field_type == DataType.NUMBER.value:
        minimum = field_def.get("minimum", 0)
        maximum = field_def.get("maximum", 1000)
        return round(random.uniform(minimum, maximum), 2)
        
    elif field_type == DataType.INTEGER.value:
        minimum = int(field_def.get("minimum", 0))
        maximum = int(field_def.get("maximum", 1000))
        return random.randint(minimum, maximum)
        
    elif field_type == DataType.BOOLEAN.value:
        return random.choice([True, False])
        
    elif field_type == DataType.DATE.value:
        return generate_random_date()
        
    # Default for unknown types
    return f"value_{index}"


def generate_formatted_value(format_type: str, field_def: Dict[str, Any]) -> Any:
    """Generates a value with a specific format."""
    if format_type == DataFormat.EMAIL.value:
        return f"user{random.randint(1, 9999)}@example.com"
        
    elif format_type == DataFormat.DATE.value:
        return generate_random_date()
        
    elif format_type == DataFormat.DATE_TIME.value:
        return f"{generate_random_date()}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z"
        
    elif format_type == DataFormat.URL.value:
        return f"https://example.com/{generate_random_string(5, 10)}"
        
    elif format_type == DataFormat.UUID.value:
        return str(uuid.uuid4())
        
    elif format_type == DataFormat.PHONE.value:
        return f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        
    # Default
    return generate_random_string(5, 15)


def generate_random_string(min_length: int = 5, max_length: int = 15) -> str:
    """Generates a random string of specified length."""
    length = random.randint(min_length, max_length)
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(chars) for _ in range(length))


def generate_random_date(
    start_date: str = "2020-01-01", 
    end_date: str = "2023-12-31"
) -> str:
    """Generates a random date between start and end dates."""
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_date = start + datetime.timedelta(days=random_days)
    
    return random_date.strftime("%Y-%m-%d")


def generate_date_sequence(
    start_date: str, 
    end_date: Optional[str] = None, 
    frequency: str = "daily", 
    size: int = 100
) -> List[str]:
    """Generates a sequence of dates with the specified frequency."""
    # Parse start date
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    
    # Determine date increment based on frequency
    if frequency.lower() in ["daily", "day"]:
        delta = datetime.timedelta(days=1)
    elif frequency.lower() in ["weekly", "week"]:
        delta = datetime.timedelta(days=7)
    elif frequency.lower() in ["monthly", "month"]:
        # Approximate month as 30 days
        delta = datetime.timedelta(days=30)
    elif frequency.lower() in ["hourly", "hour"]:
        delta = datetime.timedelta(hours=1)
    else:
        # Default to daily
        delta = datetime.timedelta(days=1)
    
    # Generate the date sequence
    dates = []
    current = start
    end = None
    
    if end_date:
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    for _ in range(size):
        dates.append(current.strftime("%Y-%m-%d"))
        current += delta
        if end and current > end:
            break
    
    return dates[:size]


def generate_timeseries_value(
    field_type: str,
    pattern: str,
    index: int,
    total_size: int,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None
) -> Union[float, int]:
    """Generates a time series value based on a pattern."""
    if minimum is None:
        minimum = 0
    if maximum is None:
        maximum = 1000
    
    range_size = maximum - minimum
    
    # Base value in the specified range
    if pattern == DataPattern.RANDOM.value:
        value = random.uniform(minimum, maximum)
    
    elif pattern == DataPattern.SEQUENTIAL.value:
        # Linear increase from min to max
        progress = index / (total_size - 1) if total_size > 1 else 0
        value = minimum + (progress * range_size)
    
    elif pattern == DataPattern.TREND_UP.value:
        # Upward trend with noise
        progress = index / (total_size - 1) if total_size > 1 else 0
        base = minimum + (progress * range_size)
        noise = random.uniform(-0.05, 0.05) * range_size
        value = base + noise
    
    elif pattern == DataPattern.TREND_DOWN.value:
        # Downward trend with noise
        progress = 1 - (index / (total_size - 1) if total_size > 1 else 0)
        base = minimum + (progress * range_size)
        noise = random.uniform(-0.05, 0.05) * range_size
        value = base + noise
    
    elif pattern == DataPattern.SEASONAL.value:
        # Seasonal pattern with period
        period = total_size / 4  # Four seasons in the dataset
        base = minimum + (range_size * 0.5)  # Center point
        amplitude = range_size * 0.4  # Amplitude of the seasonal variation
        seasonal = amplitude * math.sin(2 * math.pi * index / period)
        noise = random.uniform(-0.1, 0.1) * range_size
        value = base + seasonal + noise
    
    elif pattern == DataPattern.CYCLICAL.value:
        # Cyclical pattern
        period = total_size / 2  # Two cycles in the dataset
        base = minimum + (range_size * 0.5)
        amplitude = range_size * 0.3
        cyclical = amplitude * math.sin(2 * math.pi * index / period)
        trend = (index / total_size) * (range_size * 0.2)  # Small upward trend
        noise = random.uniform(-0.05, 0.05) * range_size
        value = base + cyclical + trend + noise
    
    else:  # Default to normal distribution
        # Normal distribution around the center of the range
        mean = minimum + (range_size / 2)
        std_dev = range_size / 6  # 99.7% of values within the range
        value = random.normalvariate(mean, std_dev)
    
    # Ensure the value stays within bounds
    value = max(minimum, min(maximum, value))
    
    # Convert to integer if that's the field type
    if field_type == DataType.INTEGER.value:
        value = int(round(value))
    else:
        value = round(value, 2)
    
    return value


def infer_schema_from_description(description: str) -> Dict[str, Any]:
    """Attempts to infer a schema from a natural language description."""
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Look for indicators of data fields
    # This is a simplistic implementation that could be enhanced with NLP
    
    # Check for time series indicators
    if any(term in description.lower() for term in ["time series", "timeseries", "over time", "temporal"]):
        schema["properties"]["date"] = {
            "type": "string",
            "format": "date",
            "description": "Date of the measurement"
        }
        schema["required"].append("date")
        
        # Look for numeric indicators
        if any(term in description.lower() for term in ["price", "cost", "value", "amount"]):
            schema["properties"]["value"] = {
                "type": "number",
                "description": "Measured value"
            }
            schema["required"].append("value")
    
    # Check for person-related data
    if any(term in description.lower() for term in ["person", "people", "individual", "customer", "user"]):
        schema["properties"]["id"] = {
            "type": "string",
            "description": "Unique identifier"
        }
        schema["required"].append("id")
        
        if "name" in description.lower():
            schema["properties"]["name"] = {
                "type": "string",
                "description": "Person's name"
            }
            schema["required"].append("name")
        
        if any(term in description.lower() for term in ["age", "years old"]):
            schema["properties"]["age"] = {
                "type": "integer",
                "minimum": 0,
                "description": "Person's age in years"
            }
    
    # If we couldn't infer anything, add a generic value field
    if len(schema["properties"]) == 0:
        schema["properties"]["id"] = {
            "type": "string",
            "description": "Unique identifier"
        }
        schema["properties"]["value"] = {
            "type": "string",
            "description": "Generic value"
        }
        schema["required"] = ["id"]
    
    return schema


def generate_suggestions(rfd: Dict[str, Any]) -> List[str]:
    """Generates suggestions to improve an RFD."""
    suggestions = []
    schema = rfd.get("schema", {})
    properties = schema.get("properties", {})
    
    # Check for missing descriptions
    for field, def_ in properties.items():
        if "description" not in def_:
            suggestions.append(f"Add a description for field '{field}'")
    
    # Check for numeric fields without constraints
    for field, def_ in properties.items():
        if def_.get("type") in ["number", "integer"] and "minimum" not in def_ and "maximum" not in def_:
            suggestions.append(f"Consider adding min/max constraints for numeric field '{field}'")
    
    # Check for string fields without format or pattern
    for field, def_ in properties.items():
        if def_.get("type") == "string" and "format" not in def_ and "pattern" not in def_:
            if field.lower() in ["email", "mail"]:
                suggestions.append(f"Consider adding email format to field '{field}'")
            elif field.lower() in ["phone", "telephone", "mobile"]:
                suggestions.append(f"Consider adding phone format to field '{field}'")
            elif "date" in field.lower() or "time" in field.lower():
                suggestions.append(f"Consider adding date format to field '{field}'")
    
    # General suggestions
    if "description" not in rfd or not rfd["description"]:
        suggestions.append("Add a detailed description to help with data generation")
    
    if len(properties) == 0:
        suggestions.append("Define at least one property in the schema")
    
    if "required" not in schema or len(schema.get("required", [])) == 0:
        suggestions.append("Specify which fields are required")
    
    return suggestions


# Import missing module
import math 