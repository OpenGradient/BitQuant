install:
	pip install -r requirements.txt

run:
	python main.py

check:
	black .
	mypy .

test:
	curl -XPOST http://127.0.0.1:5000/api/agent/run \
		-H "Content-Type: application/json" \
		-d '{"userInput":"hello","context":{"conversationHistory":[], "tokens": [], "poolPositions": [], "availablePools": []}}'	
