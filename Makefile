.PHONY: build up down logs ps health web-health llm-up llm-logs llm-pull

LLM_MODEL ?= qwen2.5:1.5b

build:
	docker compose build

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f app web chroma

llm-up:
	OLLAMA_MODEL=$(LLM_MODEL) GENERATOR_PROVIDER=ollama GENERATOR_MODEL_NAME=$(LLM_MODEL) LOCAL_MODEL_ENDPOINT=http://model:11434 docker compose --profile local-model up --build -d

llm-logs:
	docker compose logs -f app web chroma model model-init

llm-pull:
	OLLAMA_MODEL=$(LLM_MODEL) docker compose --profile local-model up model-init

ps:
	docker compose ps

health:
	curl -fsS http://localhost:$${APP_EXTERNAL_PORT:-8000}/api/health

web-health:
	curl -fsS http://localhost:$${WEB_EXTERNAL_PORT:-8010}/
