install:
	pip install -r requirements.txt

run:
	python src/main.py

test:
	pytest tests/

docker-build:
	docker build -t mlflow-app .

docker-run:
	docker run -p 5000:5000 mlflow-app
