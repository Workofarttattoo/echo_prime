# ===========================================================================
# ECH0-PRIME — Makefile
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
# All Rights Reserved. PATENT PENDING.
# ===========================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE := docker compose
IMAGE   := echo-prime
TAG     := latest

# ── Docker ────────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build Docker images
	$(COMPOSE) build

.PHONY: up
up: ## Start all services (detached)
	$(COMPOSE) up -d

.PHONY: down
down: ## Stop and remove all services
	$(COMPOSE) down

.PHONY: restart
restart: down up ## Restart all services

.PHONY: logs
logs: ## Tail logs from all services
	$(COMPOSE) logs -f --tail=100

.PHONY: ps
ps: ## Show running services
	$(COMPOSE) ps

.PHONY: shell
shell: ## Open a shell in the echo-prime container
	$(COMPOSE) exec echo-prime /bin/bash

.PHONY: shell-dashboard
shell-dashboard: ## Open a shell in the dashboard container
	$(COMPOSE) exec dashboard /bin/bash

# ── Development ───────────────────────────────────────────────────────────

.PHONY: dev
dev: ## Run the orchestrator locally (no Docker)
	ECH0_MODE=orchestrator python app.py

.PHONY: dev-dashboard
dev-dashboard: ## Run the dashboard locally (no Docker)
	ECH0_MODE=dashboard python app.py

.PHONY: dev-gradio
dev-gradio: ## Run the Gradio interface locally (no Docker)
	ECH0_MODE=gradio python app.py

# ── Quality ───────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run the test suite
	python -m pytest tests/ -v --tb=short -x

.PHONY: test-basic
test-basic: ## Run only the basic smoke tests
	python -m pytest tests/test_ech0_basic.py -v --tb=short

.PHONY: lint
lint: ## Lint with flake8
	python -m flake8 core/ reasoning/ learning/ memory/ safety/ agents/ \
		--max-line-length=120 --ignore=E501,W503,E203

.PHONY: format
format: ## Auto-format with black
	python -m black core/ reasoning/ learning/ memory/ safety/ agents/ \
		main_orchestrator.py app.py dashboard_server.py \
		--line-length=100 --target-version=py310

.PHONY: typecheck
typecheck: ## Run mypy type-checking
	python -m mypy core/ reasoning/ learning/ memory/ safety/ agents/ \
		--ignore-missing-imports --no-strict-optional

# ── Maintenance ───────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove Python caches, build artifacts, temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	find . -name '*.pyo' -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage

.PHONY: clean-docker
clean-docker: down ## Remove Docker volumes and images
	$(COMPOSE) down -v --rmi local
	docker image prune -f

.PHONY: install
install: ## Install Python dependencies locally
	pip install -r requirements.txt

.PHONY: install-dev
install-dev: install ## Install dev + prod dependencies
	pip install pytest black flake8 mypy

# ── Deploy ────────────────────────────────────────────────────────────────

.PHONY: deploy
deploy: ## Run the deployment script
	./scripts/deploy.sh

# ── Help ──────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-18s\033[0m %s\n", $$1, $$2}'
