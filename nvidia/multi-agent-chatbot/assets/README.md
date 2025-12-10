# Chatbot Spark: A Local Multi-Agent System for DGX Spark 

## Project Overview

Chatbot Spark is a fully local multi-agent system built on DGX Spark. With 128GB of unified memory, DGX Spark can run multiple LLMs and VLMs in parallel — enabling interactions across agents. 

At the core is a supervisor agent powered by GPT-OSS-120B, orchestrating specialized downstream agents for coding, retrieval-augmented generation (RAG), and image understanding. Thanks to DGX Spark’s out-of-the-box support for popular AI frameworks and libraries, development and prototyping were fast and frictionless. Together, these components demonstrate how complex, multimodal workflows can be executed efficiently on local, high-performance hardware.

> **Note**: This demo uses ~120 out of the 128GB of DGX Spark's memory by default, so ensure that no other workloads are running on your Spark using `nvidia-smi` or switch to a smaller supervisor model like gpt-oss-20B.

This project was built to be customizable, serving as a framework that developers can customize. 

## Key Features
  - **MCP Server Integration**: Chatbot Spark also showcases the ability to connect to custom MCP servers through a simple and customizable multi-server client

  - **Tool Calling**: This project uses an agents-as-tools framework and showcases the ability to create additional agents connected as tools. General tools can also be added.

  - **Easily Swappable Models**: Models are loaded and served with Ollama over the OpenAI-compatible API. Any OpenAI-compatible model available in Ollama can be integrated into the project.

  - **Vector Indexing & Retrieval**: Qdrant provides document retrieval without the jemalloc dependency.

  - **Real-time LLM Streaming**: We present custom LLM-streaming infrastructure, making it easy for developers to stream supervisor responses from any OpenAI compatible model. 

  - **gpt-oss Integration**: The default chat/tool-calling model is gpt-oss:120b, providing seamless integration with OpenAI's latest open sorce tool-calling model.


## System Overview
<img src="assets/system-diagram.png" alt="System Diagram" style="max-width:600px;border-radius:5px;justify-content:center">

## Default Models
| Model                   | Source (Ollama)     | Model Type | Notes |
|-------------------------|---------------------|------------|-------|
| gpt-oss:120b            | NVIDIA via Ollama   | Chat       | Default supervisor/chat model |
| qwen3-coder:30b         | Alibaba via Ollama  | Coding     | Used by code generation MCP tool |
| ministral-3:14b         | Mistral via Ollama  | Image      | Vision model for image understanding |
| qwen3-embedding:8b      | Alibaba via Ollama  | Embedding  | Embedding model for Qdrant |

Environment variables in `docker-compose.yml` wire these defaults to the correct agents:

- `MODELS` → chat/supervisor picker (default `gpt-oss:120b` first)
- `CODE_MODEL` → code-generation MCP tool (`qwen3-coder:30b`)
- `VISION_MODEL` → image-understanding MCP tool (`ministral-3:14b`)
- `EMBEDDING_MODEL` → vector search (`qwen3-embedding:8b`)
- `OLLAMA_AUTOSTART_MODELS` → comma-separated list of models to warm on startup (`gpt-oss:120b,qwen3-coder:30b,ministral-3:14b,qwen3-embedding:8b`)

The Ollama runtime listens on port **11434** inside and outside the Docker network. Cached models listed in `OLLAMA_AUTOSTART_MODELS` are warmed automatically when the container starts so the agents can issue requests immediately without waiting for first-token load.

> Models are pulled automatically into the Ollama volume with `./model_download.sh`.

> **Warning**:
> Since the default models use majority of available VRAM, ensure that you don't have anything already running on DGX Spark using `nvidia-smi`. If you do, switch to `gpt-oss-20b` following [this guide](#using-different-models).

---

## Quick Start
#### 1. Clone the repository and change directories to the multi-agent chatbot directory.

#### 2. Configure docker permissions
```bash
sudo usermod -aG docker $USER
newgrp docker
```

> **Warning**: After running usermod, you may need to reboot using `sudo reboot` to start a new
> session with updated group permissions.

#### 3. Run the model download script
The setup script will pull the required Ollama models into the shared volume: `gpt-oss:120b`, `qwen3-coder:30b`, `ministral-3:14b`, and `qwen3-embedding:8b`.
```bash
chmod +x model_download.sh
./model_download.sh
```

> **FLUX image generation model**: If you opt in to the FLUX image pipeline, run `scripts/download_flux_fp4.sh` from this directory. By default it writes the fp4 weights to `assets/flux-fp4` (or a custom path you pass as the first argument), which matches the backend default download location.

#### 4. Start the docker containers for the application
This step starts the Ollama runtime, Qdrant, the backend API server, and the frontend UI. This step can take 10 to 20 minutes depending on network speed.
```bash
docker compose up -d --build
```

> The compose file now builds two additional internal services:
> * **flux-service**: a GPU-enabled FLUX fp4 image generator listening on port 8080
> * **video-service**: a Wan2.2 text-to-video helper listening on port 8081

> **Tip**: The models compose file (`docker-compose-models.yml`) remains available for overrides and customization. If you prefer to keep your model configuration separate, you can still run:
> ```bash
> docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
> ```

Wait for all the containers to become ready and healthy.
```bash
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'
```

#### 5. Access the frontend UI

Open your browser and go to: [http://localhost:3000](http://localhost:3000)

> Note:  If you are running this on a remote GPU via an ssh connection, in a new terminal window, you need to run to be able to access the UI at localhost:3000 and for the UI to be able to communicate to the backend at localhost:8000:
>```bash
> ssh -L 3000:localhost:3000 -L 8000:localhost:8000  username@IP-address
>```

You should see the following UI in your browser:
<img src="assets/multi-agent-chatbot.png" alt="Frontend UI" style="max-width:600px;border-radius:5px;justify-content:center">

### Quick docker exec checks
Skip extra tooling and run these fast container checks to confirm the stack is healthy. If you skipped `./model_download.sh`, you can also pull the Ollama models directly inside the container:

```bash
# Pull the default models inside the Ollama container
docker exec -it ollama bash -lc 'for m in gpt-oss:120b qwen3-coder:30b ministral-3:14b qwen3-embedding:8b; do ollama pull "$m"; done'

# Confirm the backend responds
docker exec -it backend curl -s http://localhost:8000/health

# Verify models are visible in the Ollama runtime
docker exec -it ollama ollama list

# Spot-check Qdrant status
docker exec -it qdrant curl -s http://localhost:6333/health | jq
```

### 6. Try out the sample prompts
Click on any of the tiles on the frontend to try out the supervisor and the other agents.

#### RAG Agent:
Before trying out the example prompt for the RAG agent, upload the example PDF document [NVIDIA Blackwell Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) as context by going to the link, downloading the PDF to the local filesystem, clicking on the green "Upload Documents" button in the left sidebar under "Context" and then make sure to check the box in the "Select Sources" section.

<img src="assets/document-ingestion.png" alt="Ingest Documents" style="max-width:300px;border-radius:5px;justify-content:center">

> **Note**: You may upload any PDF of your choice, and ask corresponding queries. The default prompt requires the NVIDIA Blackwell Whitepaper.

#### Image Understanding Agent:

**Example Prompt:**

Describe this image: https://en.wikipedia.org/wiki/London_Bridge#/media/File:London_Bridge_from_St_Olaf_Stairs.jpg


---

## FLUX image and video service details

### Building and running
Both services are included in the default compose workflow. To rebuild just these two services, run:

```bash
docker compose build flux-service video-service
docker compose up -d flux-service video-service
```

> **GPU requirement**: `flux-service` requires NVIDIA GPUs. Ensure Docker has GPU access (for example via the NVIDIA Container Toolkit providing `--gpus all`). The service logs the ONNX provider it will use on startup.

### Health checks
```bash
curl http://localhost:8080/health   # FLUX service
curl http://localhost:8081/health   # Wan video service
```

The FLUX endpoint reports the ONNX provider (CUDA or CPU), model identifier, and cache path. The video endpoint reports whether the Wan model cache directory is present.

### Testing the tools directly
Generate a sample image (uses defaults inside the service):
```bash
curl -X POST http://localhost:8080/generate_image \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Create a cartoon image of a squirrel holding an elephant that is holding a weasel that is holding a beetle that is holding an ant that is holding a cupcake."}'
```

Generate a sample video (requires a valid `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` available to the container):
```bash
curl -X POST http://localhost:8081/generate_video \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Large robot throwing a a smaller robot that is carrying an even smaller robot that is wearing a dog costume."}'
```

Both endpoints return base64-encoded payloads (`image_base64` / `video_base64`) and a markdown snippet that the backend tools forward unchanged to the UI.


## Cleanup

Follow these steps to completely remove the containers and free up resources.

From the root directory of the multi-agent-chatbot project, run the following commands:

```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml down

docker volume rm "$(basename "$PWD")_postgres_data"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
You can optionally run `docker volume prune` to remove all unused volumes at the end of the demo.
> **Note**: If you do not execute these commands containers, will continue to run and take up memory.

## Customizations

### Using different models

You can use swap the model that the supervisor agent is using, for example to gpt-oss-20b.

1. In `setup.sh`, uncomment the line to download gpt-oss-20b.
> Note: If you already have the model files downloaded, you can skip to step 2.
2. In `docker-compose-models.yml`, uncomment the block for gpt-oss-20b. 
> Note: Since the default models use all of the existing VRAM, you will need to comment out the block for gpt-oss-120b in `docker-compose-models.yml`.
3. In `docker-compose.yml`, add `gpt-oss-20b` to the `MODELS` environment variable (line 40).
> Note: This name should match the container name that you set for this model in `docker-compose-models.yml`.

### Adding MCP servers and tools

1. You can add more MCP servers and tools under [backend/tools/mcp_servers](backend/tools/mcp_servers/) following existing examples.

2. If you added an MCP server, remember to add it to the server configs in [backend/client.py](backend/client.py)
