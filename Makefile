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
	docker build . -t bluefin_agent

prod:
	docker run -d -p 8000:8000 bluefin_agent

chat:
	python3.13 testclient/client.py

sample:
	curl -XPOST http://127.0.0.1:8000/api/agent/run \
	  -H "Content-Type: application/json" \
	  -d @sample-payload.json | jq

format:
	ruff format .