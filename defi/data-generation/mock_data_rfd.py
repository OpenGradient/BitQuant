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

if __name__ == "__main__":
    print("Test: generate_synthetic_data_from_rfd (BTC timeseries)")
    data = generate_synthetic_data_from_rfd.invoke({"rfd": BTC_TIMESERIES_RFD, "size": 10})
    for row in data:
        print(row)