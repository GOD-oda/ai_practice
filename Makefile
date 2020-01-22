.PHONY: build
build:
	docker-compose build

.PHONY: up
up:
	docker-compose up

.PHONY: install
install:
	docker-compose run --rm app pipenv install --three ${module}

.PHONY: destroy
destroy:
	docker-compose down --rmi all -v