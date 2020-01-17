.PHONY: build
build:
	docker-compose build

.PHONY: up
up:
	docker-compose up

.PHONY: destroy
destroy:
	docker-compose down --rmi all -v