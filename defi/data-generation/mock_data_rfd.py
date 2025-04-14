from synthetic_data_tools import generate_synthetic_data_from_rfd

BTC_TIMESERIES_RFD = {
    "rfd_id": "btc_timeseries_001",
    "name": "Synthetic BTC Time-Series Data",
    "description": "A synthetic dataset representing Bitcoin price movements over the past year, including daily open, high, low, close prices, and trading volume.",
    "schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"},
            "open_price": {"type": "number"},
            "high_price": {"type": "number"},
            "low_price": {"type": "number"},
            "close_price": {"type": "number"},
            "volume": {"type": "integer"}
        },
        "required": ["date", "open_price", "high_price", "low_price", "close_price", "volume"]
    }
}

SF_WEATHER_RFD = {
    "rfd_id": "sf_weather_may_aug_001",
    "name": "Synthetic Weather Data for San Francisco (May to August)",
    "description": "A synthetic dataset containing daily weather information for San Francisco from May to August, including temperature, humidity, and precipitation.",
    "schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"},
            "temperature": {"type": "number", "description": "Average daily temperature in degrees Fahrenheit"},
            "humidity": {"type": "number", "description": "Average daily humidity percentage"},
            "precipitation": {"type": "number", "description": "Daily precipitation in inches"}
        },
        "required": ["date", "temperature", "humidity", "precipitation"]
    }
}

HEALTHCARE_PRESCRIPTIONS_RFD = {
    "rfd_id": "healthcare_prescriptions_001",
    "name": "Synthetic Healthcare Prescription Dataset",
    "description": "A synthetic dataset detailing patient prescriptions, including patient demographics, medication details, dosage, and prescribing physician information.",
    "schema": {
        "type": "object",
        "properties": {
            "patient_id": {"type": "string"},
            "age": {"type": "integer"},
            "gender": {"type": "string", "enum": ["Male", "Female", "Other"]},
            "medication": {"type": "string"},
            "dosage": {"type": "string"},
            "prescribing_physician": {"type": "string"},
            "prescription_date": {"type": "string", "format": "date"}
        },
        "required": ["patient_id", "age", "gender", "medication", "dosage", "prescribing_physician", "prescription_date"]
    }
}

GENERIC_RFD = {
    "rfd_id": "UID",
    "name": "name",
    "description": "A request for a dataset with a specified schema.",
    "schema": {
        "type": "object",
        "properties": {
            "field1": {"type": "string", "description": "Example field representing textual data."},
            "field2": {"type": "number", "description": "Example field representing numerical data."},
            "field3": {"type": "boolean", "description": "Example field representing a boolean value."}
        },
        "required": ["field1", "field2"]
    }
}

if __name__ == "__main__":
    print("Test: generate_synthetic_data_from_rfd (BTC timeseries)")
    data = generate_synthetic_data_from_rfd.invoke({"rfd": BTC_TIMESERIES_RFD, "size": 3})
    for row in data:
        print(row)
    print("\nTest: generate_synthetic_data_from_rfd (SF Weather)")
    data = generate_synthetic_data_from_rfd.invoke({"rfd": SF_WEATHER_RFD, "size": 3})
    for row in data:
        print(row)
    print("\nTest: generate_synthetic_data_from_rfd (Healthcare Prescriptions)")
    data = generate_synthetic_data_from_rfd.invoke({"rfd": HEALTHCARE_PRESCRIPTIONS_RFD, "size": 3})
    for row in data:
        print(row)
    print("\nTest: generate_synthetic_data_from_rfd (Generic RFD)")
    data = generate_synthetic_data_from_rfd.invoke({"rfd": GENERIC_RFD, "size": 3})
    for row in data:
        print(row)