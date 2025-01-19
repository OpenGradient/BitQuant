install:
	pip install -r requirements.txt

run:
	python main.py

check:
	black .
	mypy .


define JSON_DATA
{
	"userInput": "i wanna withdraw everything",
	"context": {
		"conversationHistory": [],
		"tokens": [
			{"amount": 100, "symbol": "SUI"},
			{"amount": 45333, "symbol": "USDC"}
		],
		"poolPositions": [
			{"poolSymbol": "SUI/USDC", "amountDeposited": 10000}
		],
		"availablePools": [{
			"address": "0x123",
			"symbol": "SUI/USDC",
			"tokenA": "SUI",
			"tokenB": "USDC",
			"TVL": "100M USD",
			"APRLastDay": 12,
			"APRLastWeek": 8,
			"APRLastMonth": 4
		}]
	}
}
endef
export JSON_DATA

test:
	curl -XPOST http://127.0.0.1:5000/api/agent/run \
		-H "Content-Type: application/json" \
		-d "$${JSON_DATA}"
