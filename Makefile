PYTHON ?= python

.PHONY: build test bench clean

build:
	$(PYTHON) -m mojoscf.build --force

test: build
	$(PYTHON) -m pytest -q

bench: build
	$(PYTHON) benchmarks/bench_scf.py

clean:
	rm -f mojoscf/_mojoscf.so mojoscf/_mojoscf.hash mojoscf/_mojoscf.building.so
	rm -rf mojoscf/__pycache__ tests/__pycache__ .pytest_cache
