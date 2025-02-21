venv:
	python3 -m venv venv
	@echo "To activate the virtual environment, run: source venv/bin/activate"

install:
	pip install -r requirements.txt

run:
	python main.py

check:
	black .
	mypy .

integration-test:
	python -m unittest server/test_server.py

test:
	python -m unittest discover

docker:
	docker build . -t bluefin_agent

prod:
	docker run -d -p 8000:8000 bluefin_agent

sample:
	curl -XPOST http://127.0.0.1:5000/api/agent/run \
	  -H "Content-Type: application/json" \
	  -d @sample-payload.json | jq