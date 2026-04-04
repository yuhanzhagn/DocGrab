.PHONY: build up down logs ps health

build:
	docker compose build

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f app chroma

ps:
	docker compose ps

health:
	curl -fsS http://localhost:$${APP_EXTERNAL_PORT:-8000}/api/health
