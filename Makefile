install:
	pip install -r requirements.txt

run:
	python main.py

format:
	black .

test:
	curl -XPOST http://127.0.0.1:5000/api/agent/run \
		-H "Content-Type: application/json" \
		-d '{}'	
