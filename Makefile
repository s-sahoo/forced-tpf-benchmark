PYTHON ?= python3
FIGURES := 1 2 3 4

.PHONY: figures validate-paper-data

validate-paper-data:
	$(PYTHON) configs/validate_forced_tpf_calibration.py
	$(PYTHON) paper/validate_data.py

figures: validate-paper-data
	@set -e; for figure in $(FIGURES); do \
		$(PYTHON) paper/figure$$figure/plot.py; \
	done
