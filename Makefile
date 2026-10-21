DOCKER_IMAGE ?= zeroerpp
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOCKER_PARAMETERS := \
	--user "$(shell id -u):$(shell id -g)" \
	-v "$(ROOT):/app" \
	-w /app \
	-e PYTHON=python3

.PHONY: init run_module .require-module

init:
	docker build -t "$(DOCKER_IMAGE)" "$(ROOT)"

run_module: .require-module
	docker run -i --rm $(DOCKER_PARAMETERS) "$(DOCKER_IMAGE)" $(module)

.require-module:
ifndef module
	$(error module is required, for example module="bash scripts/reproduce.sh")
endif
