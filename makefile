


PYTHON = python
PIP = pip


.PHONY: install
install:
	$(PIP) install black torch transformers Pillow

.PHONY: format
format:
	black .

.PHONY: all
all: install format

