DOMAIN = checkconnect
SRC_DIR = src
LOCALES_DIR = $(SRC_DIR)/$(DOMAIN)/locales
GUI_DIR = $(SRC_DIR)/$(DOMAIN)/gui
TRANSLATIONS_DIR = $(GUI_DIR)/translations
LANGUAGES = en es fr de it pt pt_BR ru zh_CN zh_TW ja ko hi tr ar vi pl nl sv


.PHONY: build qt babel all test coverage lint docs-build docs-lint pre-commit clean help

# Default target
all: qt babel

# Example target to print the variables
print:
	@echo "Locales Directory: $(LOCALES_DIR)"
	@echo "GUI Directory: $(GUI_DIR)"
	@echo "Translations Directory: $(TRANSLATIONS_DIR)"
	@echo "Languages: $(LANGUAGES)"

lint-deps:
	@echo "Installing linting tools..."
	hatch shell lint || true

pyright: lint-deps
	@echo "Run pyright..."
	hatch run lint:typing --all

pylint: lint-deps
	@echo "Run pylint..."
	hatch run lint:lint --all

codespell: lint-deps
	@echo "Run codespell..."
	hatch run lint:spelling --all


install_prettier:
	npm install -g prettier

ruff:
	-ruff check --fix .
	ruff format .

check_ruff:
	ruff check .
	ruff format --check .

# Target to update Qt translations
qt:
	@echo "Updating Qt translations..."
	@for lang in $(LANGUAGES); do \
		ts_file=$(TRANSLATIONS_DIR)/$$lang.ts; \
		qm_file=$(TRANSLATIONS_DIR)/$$lang.qm; \
		if [ ! -f "$$ts_file" ]; then \
			echo '<?xml version="1.0" encoding="utf-8"?><!DOCTYPE TS><TS version="2.1" language="'"$$lang"'"></TS>' > "$$ts_file"; \
		fi; \
		pyside6-lupdate $(shell find $(GUI_DIR) -name '*.py') -ts $$ts_file; \
		pyside6-lrelease $$ts_file -qm $$qm_file; \
	done; \
	pyside6-rcc $(GUI_DIR)/resources.qrc -o $(GUI_DIR)/resources_rc.py

install-babel:
	@echo "Installing Babel..."
	hatch shell translate || true

babel: install-babel
	@echo "Updating Babel translations..."
	@for lang in $(LANGUAGES); do \
		po_dir=$(LOCALES_DIR)/$$lang/LC_MESSAGES; \
		mkdir -p $$po_dir; \
		po_file=$$po_dir/messages.po; \
		pot_file=$(LOCALES_DIR)/$(DOMAIN).pot; \
		pybabel extract -k translate -k _translate_func -o $$pot_file $(shell find $(SRC_DIR) -name '*.py'); \
		if [ -f $$po_file ]; then \
			pybabel update -i $$pot_file -d $$po_dir -l $$lang; \
		else \
			pybabel init -i $$pot_file -d $$po_dir -l $$lang; \
		fi; \
		pybabel compile -d $$po_dir -l $$lang; \
	done

# Run build
build:
	@echo "Build..."
	hatch build

# Run tests using Hatch
test:
	@echo "Running tests..."
	hatch run test:test

# Run coverage
coverage:
	@echo "Run coverage"
	hatch run test:coverage

# Lint the code using Hatch
lint: lint-deps
	@echo "Linting the code..."
	hatch run lint:all .
	hatch run security:all

docs-build:
	@echo "Build docs"
	hatch run docs:build

docs-lint: docs-build
	@echo "Lint docs"
	hatch run docs:validate-links
	hatch run docs:qualitycheck
	hatch run docs: serve

pre-commit: lint-deps
	@echo "Run pre-commit"
	hatch run lint:precommit

# Clean up generated files
clean:
	@echo "Cleaning up the builds"
	@rm -rf dist
	@echo "Cleaning up the docs"
	@rm -rf site
	@echo "Cleaning up translation files..."
	@rm -rf $(LOCALES_DIR)/*/LC_MESSAGES/*.mo
	@rm -rf $(LOCALES_DIR)/*/LC_MESSAGES/*.po
	@echo "Cleaning up Qt translation files..."
	@rm -rf $(TRANSLATIONS_DIR)/*.ts
	@rm -rf $(TRANSLATIONS_DIR)/*.qm
	@rm -f $(GUI_DIR)/resources_rc.py
	@echo "Cleaning up caches"
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@rm -rf .cache

# Help target to display available commands
help:
	@echo "Makefile commands:"
	@echo "  all      - Update Qt and Babel translations"
	@echo "  qt       - Update Qt translations"
	@echo "  babel    - Update Babel translations"
	@echo "  test     - Run tests"
	@echo "  lint     - Lint the code"
	@echo "  clean    - Clean up generated files"
	@echo "  help     - Display this help message"
