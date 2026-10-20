PYTHON ?= python3
ARGS ?=
export PYTHON

.PHONY: init init_local dataset verify-data prepare blocking primary sensitivity match tune baselines timings report reproduce

init: docker_build

init_local:
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

tune:
	bash scripts/tune.sh $(ARGS)

baselines:
	bash scripts/baselines.sh $(ARGS)

timings:
	bash scripts/timings.sh $(ARGS)

report:
	bash scripts/report.sh $(ARGS)

reproduce:
	bash scripts/reproduce.sh $(ARGS)

# The image contains dependencies and runs the mounted checkout.
DOCKER_IMAGE ?= zeroerpp
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKER_PARAMETERS := \
	--user "$(shell id -u):$(shell id -g)" \
	-v "$(ROOT):/app" \
	-w /app \
	-e PYTHON=python3

.PHONY: docker_build run_module bash_docker .require-module

docker_build:
	docker build -t "$(DOCKER_IMAGE)" "$(ROOT)"

run_module: .require-module
	docker run -i --rm $(DOCKER_PARAMETERS) "$(DOCKER_IMAGE)" $(module)

bash_docker:
	docker run -it --rm $(DOCKER_PARAMETERS) "$(DOCKER_IMAGE)" bash

.require-module:
ifndef module
	$(error module is required, for example module="bash scripts/reproduce.sh")
endif
