# AI Customer Support Agent

This project will provide an AI-powered customer support agent that can answer customer questions accurately, use business knowledge and support tools, and offer a reliable foundation for a production-ready support experience.

## Planned Architecture

The planned architecture separates the agent workflow, retrieval-augmented generation, tools, database access, API, configuration, and evaluation into focused modules.

## Current Status

The project currently has a minimal LangGraph agent available through a CLI and FastAPI HTTP API, with Redis-backed conversation checkpoints, PostgreSQL tools, human-approved order cancellation, S3-sourced policy ingestion into Chroma, and a small local evaluation suite.

## Docker

Build the application image from the repository root:

```bash
docker build -t customer-support-agent .
```

Before running the container, build the local Chroma index and make sure `.env` contains connection URLs that are reachable from inside Docker. If PostgreSQL and Redis run on the host with Docker Desktop, use `host.docker.internal` instead of `localhost` in those URLs.

Run the API container with runtime environment variables and the local Chroma index:

```bash
docker run --rm \
  --name customer-support-agent \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -p 8000:8000 \
  -v "$(pwd)/data/processed/chroma:/app/data/processed/chroma" \
  customer-support-agent
```

PostgreSQL and Redis must run separately or be available through network-accessible service addresses. The `.env` file is supplied only when the container starts and is not included in the image.
