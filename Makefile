install:
	pip install -r requirements.txt

run:
	python main.py

check:
	black .
	mypy .

test:
	python -m unittest server/test_server.py
