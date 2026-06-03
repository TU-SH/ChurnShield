.PHONY: help setup data db mlflow train serve dashboard up down test lint clean

help:
	@echo ""
	@echo "ChurnShield — Available commands:"
	@echo "  make setup      Install all dependencies"
	@echo "  make data       Generate synthetic AU customer data"
	@echo "  make db         Start PostgreSQL + init schema"
	@echo "  make mlflow     Start MLflow tracking server"
	@echo "  make train      Train XGBoost model + log to MLflow"
	@echo "  make serve      Start FastAPI (port 8000)"
	@echo "  make dashboard  Start Streamlit dashboard (port 8501)"
	@echo "  make up         Start everything with Docker Compose"
	@echo "  make test       Run all tests with coverage"
	@echo "  make lint       Lint and format code"
	@echo "  make clean      Stop and remove all Docker containers"
	@echo ""

setup:
	pip install -r requirements.txt -r requirements-dev.txt
	cp -n .env.example .env || true
	@echo "Setup complete. Edit .env if needed."

data:
	python generate_data.py

db:
	docker compose up postgres -d
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 4
	@echo "PostgreSQL ready at localhost:5432"

mlflow:
	docker compose up mlflow -d
	@echo "MLflow UI: http://localhost:5000"

train:
	python -m src.models.train
	@echo "Training complete. View at http://localhost:5000"

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
	@echo "API docs: http://localhost:8000/docs"

dashboard:
	streamlit run dashboard/app.py --server.port 8501

up:
	docker compose up -d
	@echo ""
	@echo "Services starting..."
	@echo "  PostgreSQL  : localhost:5432"
	@echo "  MLflow UI   : http://localhost:5000"
	@echo "  API docs    : http://localhost:8000/docs"
	@echo "  Dashboard   : http://localhost:8501"

down:
	docker compose down

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/test_features.py tests/test_model.py -v

lint:
	ruff check src/ tests/ --fix
	black src/ tests/

clean:
	docker compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f mlruns_local_model_*.pkl
	@echo "Clean complete."
