venv:
	python3 -m venv venv
	@echo "To activate the virtual environment, run: source venv/bin/activate"

install:
	pip install -r requirements.txt

run:
	python3.13 main.py

check:
	black .
	mypy .

integration-test:
	python3.13 -m unittest server/test_server.py

test:
	python3.13 -m unittest discover

docker:
	docker build . -t bitquant_agent

prod:
	docker run -d -p 8000:8000 bitquant_agent

chat:
	python3.13 testclient/client.py

sample:
	@echo "Set FIREBASE_ID_TOKEN or SKIP_TOKEN_AUTH_HEADER/SKIP_TOKEN_AUTH_KEY before running make sample."
	curl -XPOST http://127.0.0.1:8000/api/v2/agent/run \
	  -H "Content-Type: application/json" \
	  $(if $(FIREBASE_ID_TOKEN),-H "Authorization: Bearer $(FIREBASE_ID_TOKEN)") \
	  $(if $(and $(SKIP_TOKEN_AUTH_HEADER),$(SKIP_TOKEN_AUTH_KEY)),-H "$(SKIP_TOKEN_AUTH_HEADER): $(SKIP_TOKEN_AUTH_KEY)") \
	  -d @sample-payload.json | jq

format:
	ruff format .
