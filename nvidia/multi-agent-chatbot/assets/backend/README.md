# Backend

FastAPI Python application serving as the API backend for the chatbot demo.

## Overview

The backend handles:
- Multi-model LLM integration (local models)
- Document ingestion and vector storage for RAG
- WebSocket connections for real-time chat streaming
- Image processing and analysis
- Chat history management
- Model Control Protocol (MCP) integration

> **Note on MCP servers**
> The MCP client (see [client.py](client.py)) still starts and connects to the configured MCP servers; the recent shift from
> NIMs/ArangoDB to Ollama-hosted models does not change the MCP server lifecycle or tool discovery. The WebSocket/chat changes
> only affect how the frontend reaches the FastAPI backend and do not alter MCP tooling.

## Key Features

- **Multi-model support**: Integrates various LLM providers and local models
- **RAG pipeline**: Document processing, embedding generation, and retrieval
- **Streaming responses**: Real-time token streaming via WebSocket
- **Image analysis**: Multi-modal capabilities for image understanding
- **Vector database**: Efficient similarity search for document retrieval
- **Session management**: Chat history and context persistence
- **Self-authored tooling**: The LLM can mint new MCP tools on the fly and
  persist them under `self_tooling/` with guardrails against destructive
  commands. The `self-tooling-server` MCP service exposes create/list/run APIs
  so the assistant can safely add utilities like SSH setup helpers.

## Architecture

FastAPI application with async support, integrated with vector databases for RAG functionality and WebSocket endpoints for real-time communication.

## Docker Troubleshooting

### Container Issues
- **Port conflicts**: Ensure port 8000 is not in use
- **Memory issues**: Backend requires significant RAM for model loading
- **Startup failures**: Check if required environment variables are set

### Model Loading Problems
```bash
# Check model download status
docker logs backend | grep -i "model"

# Verify model files exist
docker exec -it cbackend ls -la /app/models/

# Check available disk space
docker exec -it backend df -h
```

### Common Commands
```bash
# View backend logs
docker logs -f backend

# Restart backend container
docker restart backend

# Rebuild backend
docker-compose up --build -d backend

# Access container shell
docker exec -it backend /bin/bash

# Check API health
curl http://localhost:8000/health
```

### Performance Issues
- **Slow responses**: Check GPU availability and model size
- **Memory errors**: Increase Docker memory limit or use smaller models
- **Connection timeouts**: Verify WebSocket connections and firewall settings

### RAG and embedding startup
The vector store now performs a quick health check before trying to talk to the embedding
endpoint. If your embedding service takes longer to come online, you can adjust the
defaults via environment variables:

- `EMBEDDING_HEALTH_RETRIES` / `EMBEDDING_HEALTH_TIMEOUT`: Control how many fast health
  probes are attempted and how long each waits.
- `EMBEDDING_INIT_RETRIES` / `EMBEDDING_INIT_BACKOFF`: Control how long to wait for the
  embedding model to answer the initial dimension probe.
- `EMBEDDING_DIMENSIONS`: Skip probing entirely by supplying a known embedding dimension
  for your model (useful when you want the API to start immediately).

## Code generation configuration

To keep the MCP code generation spoke on the local Ollama network endpoint, set the
following in your environment:

```
CODEGEN_PROVIDER=ollama
OLLAMA_OPENAI_BASE_URL=http://ollama:11434/v1
CODEGEN_MODEL=gpt-oss:120b
```
