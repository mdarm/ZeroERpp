PYTHON ?= python3
ARGS ?=
export PYTHON

.PHONY: init dataset verify-data prepare blocking primary sensitivity match baselines timings report reproduce

init:
	$(PYTHON) -m pip install -r requirements.txt

dataset:
	$(PYTHON) -u main_zeroer.py download $(ARGS)

verify-data:
	$(PYTHON) -u main_zeroer.py verify-data $(ARGS)

prepare:
	$(PYTHON) -u main_zeroer.py prepare --scope all $(ARGS)

blocking:
	bash scripts/blocking.sh $(ARGS)

primary:
	bash scripts/primary.sh $(ARGS)

sensitivity:
	bash scripts/sensitivity.sh $(ARGS)

match:
	$(PYTHON) -u main_zeroer.py match --scope all --workers 4 $(ARGS)

baselines:
	bash scripts/baselines.sh $(ARGS)

timings:
	bash scripts/timings.sh $(ARGS)

report:
	bash scripts/report.sh $(ARGS)

reproduce:
	bash scripts/reproduce.sh $(ARGS)
