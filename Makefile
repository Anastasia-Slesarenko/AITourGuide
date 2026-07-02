.PHONY: help install install-dev test test-unit test-integration lint format clean run docker-build docker-up docker-down docker-pull build-index download-models

# ========================================
# HELP
# ========================================
help:
	@echo "AITourGuide - Makefile Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make download-models  Download required models"
	@echo "  make build-index      Build FAISS index"
	@echo ""
	@echo "Development:"
	@echo "  make run              Run development server"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make load-test        Run load test with Locust (headless)"
	@echo "  make lint             Run linters"
	@echo "  make format           Format code with ruff"
	@echo "  make clean            Clean temporary files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image locally"
	@echo "  make docker-pull      Pull latest API image from ghcr.io, then start"
	@echo "  make docker-up        Start Docker containers (build if image missing)"
	@echo "  make docker-down      Stop Docker containers"
	@echo ""

# ========================================
# INSTALLATION
# ========================================
install:
	@echo "📦 Installing production dependencies..."
	pip install --upgrade pip
	pip install -r requirements-prod.txt

install-dev:
	@echo "📦 Installing development dependencies..."
	pip install --upgrade pip
	pip install -r requirements-prod.txt
	pip install -r requirements-dev.txt

# ========================================
# MODELS & DATA
# ========================================
download-models:
	@echo "📥 Downloading models..."
	python scripts/setup/download_siglip_model.py
	@echo "⚠️  Note: VLM models need to be downloaded manually"
	@echo "   See README.md for instructions"

build-index:
	@echo "🔨 Building FAISS index..."
	python scripts/data_preparation/step6_setup_dataset.py

# ========================================
# DEVELOPMENT
# ========================================
run:
	@echo "🚀 Starting development server..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	@echo "🚀 Starting production server..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

# ========================================
# TESTING
# ========================================
test:
	@echo "🧪 Running all tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v -m "not slow"

test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

test-fast:
	@echo "🧪 Running fast tests only..."
	pytest tests/ -v -m "not slow and not integration"

test-coverage:
	@echo "📊 Generating coverage report..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

load-test:
	@echo "🔥 Running load test (100 users, 10/s spawn rate, 60s)..."
	@mkdir -p tests/load/results
	locust -f tests/load/locustfile.py \
		--host http://localhost:8000 \
		--headless -u 100 -r 10 -t 60s \
		--csv tests/load/results/report \
		--html tests/load/results/report.html
	@echo "✅ Results saved to tests/load/results/"

# ========================================
# CODE QUALITY
# ========================================
lint:
	@echo "🔍 Running linters..."
	@echo "  → ruff..."
	ruff check src/ tests/
	@echo "  → mypy..."
	mypy src/

format:
	@echo "✨ Formatting code..."
	ruff check --fix src/ tests/
	ruff format src/ tests/
	@echo "✅ Code formatted!"

format-check:
	@echo "🔍 Checking code formatting..."
	ruff check src/ tests/
	ruff format --check src/ tests/

# ========================================
# CLEANUP
# ========================================
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf build dist
	rm -rf .DS_Store
	@echo "✅ Cleanup complete!"

clean-data:
	@echo "⚠️  Cleaning data files (use with caution)..."
	rm -rf data/indices/*.faiss
	rm -rf data/indices/*.pkl
	@echo "✅ Data cleaned!"

# ========================================
# DOCKER
# ========================================
docker-build:
	@echo "🐳 Building Docker image..."
	docker-compose -f docker/docker-compose.yml build

docker-up:
	@echo "🐳 Starting Docker containers..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✅ Containers started!"
	@echo "   API: http://localhost:8000"
	@echo "   Docs: http://localhost:8000/docs"

docker-pull:
	@echo "🐳 Pulling latest API image from ghcr.io..."
	docker-compose -f docker/docker-compose.yml pull api
	@echo "🐳 Starting Docker containers..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✅ Containers started!"
	@echo "   API: http://localhost:8000"
	@echo "   Docs: http://localhost:8000/docs"

docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	@echo "📋 Showing Docker logs..."
	docker-compose -f docker/docker-compose.yml logs -f

docker-shell:
	@echo "🐚 Opening shell in container..."
	docker-compose -f docker/docker-compose.yml exec api /bin/bash

# ========================================
# UTILITIES
# ========================================
check-env:
	@echo "🔍 Checking environment..."
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

health-check:
	@echo "🏥 Checking API health..."
	@curl -s http://localhost:8000/v1/health | python -m json.tool || echo "❌ API not responding"

# ========================================
# DEVELOPMENT HELPERS
# ========================================
setup-dev: install-dev
	@echo "🔧 Setting up development environment..."
	cp .env.example .env
	@echo "✅ Development environment ready!"
	@echo "   Edit .env with your API keys"
	@echo "   Run 'make download-models' to download models"
	@echo "   Run 'make build-index' to build FAISS index"
	@echo "   Run 'make run' to start development server"

update-deps:
	@echo "📦 Updating dependencies..."
	pip install --upgrade pip
	pip install --upgrade -r requirements-prod.txt
	pip install --upgrade -r requirements-dev.txt

freeze-deps:
	@echo "📦 Freezing dependencies..."
	pip freeze > requirements.lock

# ========================================
# DOCUMENTATION
# ========================================
docs-serve:
	@echo "📚 Serving documentation..."
	@echo "⚠️  Documentation server not configured yet"
	@echo "   Visit http://localhost:8000/docs for API docs"
