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
