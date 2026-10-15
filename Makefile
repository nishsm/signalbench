CONFIG ?= configs/default.yaml
RUN = python -m signalbench.cli --config $(CONFIG)
.PHONY: data bench finetune signal backtest report all test edgar prices publish
data:
	$(RUN) data
bench:
	$(RUN) bench
finetune:
	$(RUN) finetune
edgar:
	$(RUN) edgar
prices:
	$(RUN) prices
signal:
	$(RUN) signal
backtest:
	$(RUN) backtest
report:
	$(RUN) report
publish:
	$(RUN) publish
all: data bench signal backtest report
test:
	python -m pytest -q

universe:
	$(RUN) universe
