SHELL := /bin/bash
DKC := $(shell command -v docker-compose 2> /dev/null || { command -v docker > /dev/null && echo 'docker compose'; })

ifeq ($(DKC),)
$(error Neither docker-compose nor docker compose found)
endif

export USERID ?= $(shell id -u)
export GROUPID ?= $(shell id -g)

.PHONY: up db-refresh sentiment-topic comments down build clean

up: db-refresh
	export MODE=run && $(DKC) up -d --remove-orphans streamlit 

db-refresh: down build
	export MODE=refresh && $(DKC) run --remove-orphans --rm streamlit

sentiment-topic: down build
	$(DKC) run --remove-orphans --rm sentiment-topic

comments: down build
	$(DKC) run --remove-orphans --rm comments

down:
	$(DKC) down
	$(DKC) rm -fsv

build:
	$(DKC) build

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
