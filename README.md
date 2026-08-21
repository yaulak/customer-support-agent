# AI Customer Support Agent

This project will provide an AI-powered customer support agent that can answer customer questions accurately, use business knowledge and support tools, and offer a reliable foundation for a production-ready support experience.

## Planned Architecture

The planned architecture separates the agent workflow, retrieval-augmented generation, tools, database access, API, configuration, and evaluation into focused modules.

## Current Status

The project currently has a minimal LangGraph agent available through a CLI and FastAPI HTTP API, with Redis-backed conversation checkpoints, PostgreSQL tools, human-approved order cancellation, local policy retrieval through Chroma, and a small local evaluation suite.
