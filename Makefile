install:
	pip install -r requirements.txt

run:
	python main.py

check:
	black .
	mypy .

test:
	python -m unittest server/test_server.py

docker:
	docker build . -t bluefin_agent

prod:
	docker run -d -p 8000:8000 --env-file .env bluefin_agent
