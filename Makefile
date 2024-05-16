SHELL := /bin/bash

.PHONY: up db-refresh sentiment-topic comments down build clean

up: db-refresh
	export MODE=run && docker-compose up -d --remove-orphans streamlit 

db-refresh: down build
	export MODE=refresh && docker-compose run --remove-orphans --rm streamlit

sentiment-topic: down build
	docker-compose run --remove-orphans --rm sentiment-topic

comments: down build
	docker-compose run --remove-orphans --rm comments

down:
	docker-compose down
	docker-compose rm -fsv

build:
	docker-compose build

clean:
	rm -rf build
	rm -rf dist
	find . -name '__pycache__' -not -path '*/\.git/*' -exec rm -rf {} \+
	find . -name '.pytest_cache' -not -path '*/\.git/*' -exec rm -rf {} \+
	find . -name '.mypy_cache' -not -path '*/\.git/*' -exec rm -rf {} \+
	find . -name '*.pyc'       -not -path '*/\.git/*' -exec rm -f {} \+
	find . -name '*.pyo'       -not -path '*/\.git/*' -exec rm -f {} \+
	find . -name '*.egg-info'  -not -path '*/\.git/*' -exec rm -rf {} \+
	find . -name '*~'          -not -path '*/\.git/*' -exec rm -f {} \+
	find . -name tags          -not -path '*/\.git/*' -exec rm -f {} \+
	find . -name tags.lock     -not -path '*/\.git/*' -exec rm -f {} \+

