install:
	pip install -r requirements.txt

run:
	python main.py

check:
	black .
	mypy .


define JSON_DATA
{
	"userInput": "what can i do?",
	"context": {
		"conversationHistory": [],
		"tokens": [],
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
