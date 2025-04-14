from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
import random
import datetime


def generate_random_string(min_length: int = 5, max_length: int = 15) -> str:
    length = random.randint(min_length, max_length)
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(chars) for _ in range(length))

def generate_random_date(start_date: str = "2020-01-01", end_date: str = "2023-12-31") -> str:
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_date = start + datetime.timedelta(days=random_days)
    return random_date.strftime("%Y-%m-%d")


def _generate_value(field_def: dict) -> Any:
    t = field_def.get("type")
    if "enum" in field_def:
        return random.choice(field_def["enum"])
    if t == "string":
        if field_def.get("format") == "date":
            return generate_random_date()
        return generate_random_string()
    if t == "number":
        minimum = field_def.get("minimum", 0)
        maximum = field_def.get("maximum", 100000)
        return round(random.uniform(minimum, maximum), 2)
    if t == "integer":
        minimum = int(field_def.get("minimum", 0))
        maximum = int(field_def.get("maximum", 100000))
        return random.randint(minimum, maximum)
    if t == "boolean":
        return random.choice([True, False])
    return None

@tool()
def generate_synthetic_data_from_rfd(rfd: Dict[str, Any], size: int = 10) -> List[Dict[str, Any]]:
    """
    Generate synthetic data from a Request For Data (RFD) JSON object and size.
    The RFD should have a 'schema' key specifying 'properties' and 'required' fields.
    Returns a list of dicts matching the schema.
    """
    schema = rfd["schema"]
    properties = schema["properties"]
    required = schema.get("required", list(properties.keys()))
    data = []
    for _ in range(size):
        row = {}
        for field in required:
            row[field] = _generate_value(properties[field])
        data.append(row)
    return data
