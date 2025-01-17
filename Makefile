install:
	pip install -r requirements.txt

run:
	python main.py

check:
	black .
	mypy .


define JSON_DATA
{
	"userInput": "deposit into the best pool",
	"context": {
		"conversationHistory": [],
		"tokens": [
			{"amount": 100, "symbol": "SUI"},
			{"amount": 45333, "symbol": "USDC"}
		],
		"poolPositions": [],
		"availablePools": [{
			"address": "0x123",
			"symbol": "SUI/USDC",
			"tvl": "100M USD",
			"tokenA": "SUI",
			"tokenB": "USDC",
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
