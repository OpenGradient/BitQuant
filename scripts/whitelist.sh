#!/bin/bash

# API configuration
API_KEY=$WHITELIST_API_KEY
API_URL="http://localhost:5000/api/whitelist/add"

# Check if the CSV file exists
if [ ! -f "scripts/users.csv" ]; then
    echo "Error: users.csv file not found"
    exit 1
fi

# Process rows between 60,000 and 85,000
tail -n +60001 scripts/users.csv | head -n 25000 | while IFS=, read -r timestamp email wallet country experience profession; do
    # Remove any quotes from the wallet address
    wallet=$(echo "$wallet" | tr -d '"')
    
    # Check if the wallet address is not empty
    if [ ! -z "$wallet" ]; then
        echo "Whitelisting wallet: $wallet"
        
        # Make the API request
        response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL" \
            -H "X-API-Key: $API_KEY" \
            -H "Content-Type: application/json" \
            --data "{\"address\":\"$wallet\"}")
        
        # Extract the status code and response body
        status_code=$(echo "$response" | tail -n1)
        response_body=$(echo "$response" | sed '$d')
        
        # Check if the request was successful (status code 200)
        if [ "$status_code" -eq 200 ]; then
            echo "Successfully whitelisted $wallet"
        else
            echo "Failed to whitelist $wallet"
            echo "Response: $response_body"
        fi
    fi
done

echo "Whitelisting process completed"