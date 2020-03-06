.PHONY: build
build:
	docker-compose build

.PHONY: up
up:
	docker-compose up

.PHONY: run
run:
	docker-compose run --rm app

.PHONY: destroy
destroy:
	docker-compose down --rmi all -v

.PHONY: pipenv_install
pipenv_install:
	docker-compose run --rm app pipenv install