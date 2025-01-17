install:
	pip install -r requirements.txt

run:
	python main.py

check:
	black .
	mypy .


define JSON_DATA
{
	"userInput": "what tokens do i have?",
	"context": {
		"conversationHistory": [],
		"tokens": [{"amount": 100, "asset": {"name": "SUI"}}],
		"poolPositions": [],
		"availablePools": []
	}
}
endef
export JSON_DATA

test:
	curl -XPOST http://127.0.0.1:5000/api/agent/run \
		-H "Content-Type: application/json" \
		-d "$${JSON_DATA}"
