


PYTHON = python
PIP = pip


.PHONY: install
install:
	$(PIP) install black

.PHONY: format
format:
	black .

.PHONY: all
all: install format
