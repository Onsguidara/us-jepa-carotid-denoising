# US-JEPA Carotid Denoising — n8n Agent

An n8n-based conversational agent that denoises carotid ultrasound images and explains the results, backed by a FastAPI service running the US-JEPA denoising model.

## How it works

1. **Webhook** (`POST /webhook/denoising`) receives a request containing an image (binary) and/or an `image_url`.
2. **Edit Fields** sets a default chat message asking the agent to analyze and explain the denoising result.
3. **Upload Image** forwards the binary image to the backend (`POST http://host.docker.internal:8000/upload`) which returns an `image_url`.
4. **Build Agent Input** composes the `chatInput` text, embedding the uploaded image URL if present.
5. **AI Agent** (LLM: OpenRouter, `openrouter/free`) receives the message. It has access to one tool:
   - **denoise_ultrasound_image** — calls `POST http://host.docker.internal:8000/denoise` with `image_url`, returns the denoised image (base64) and no-reference quality metrics (noise proxy, edge energy, inference time). The agent is instructed to call this tool whenever an image + denoising request is present, and never invent metrics.
6. **Respond to Webhook** returns an HTML page showing the denoised image and the metrics/explanation.

## Structure

```
agent/
├── denoising_agent_n8n.json   # n8n workflow export (import this into n8n)
├── main.py                    # FastAPI backend: /upload and /denoise endpoints
├── models/                    # US-JEPA model weights / architecture used for denoising
├── requirements.txt           # Python dependencies for the backend
├── test.jpg                   # sample ultrasound image for testing
└── storage/                   # runtime storage for uploaded/processed images (gitignored)
```

## Setup

### 1. Start n8n (Docker)

```bash
cd C:\Users\User\n8n
dir /a
docker compose up -d
```

n8n will be available at [http://localhost:5678](http://localhost:5678).

### 2. Start the FastAPI backend

```bash
cd agent
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Expected endpoints:
- `POST /upload` — accepts a binary file, returns `{ "image_url": "..." }`
- `POST /denoise` — accepts `{ "image_url": "..." }`, returns `{ "denoised_image_base64": "...", "metrics": { "noise_proxy": ..., "edge_energy": ..., "inference_time_sec": ... } }`

n8n reaches this backend via `http://host.docker.internal:8000` (already set in the workflow nodes).

### 3. Import and activate the n8n Workflow

1. Import `denoising_agent_n8n.json` into n8n via **Workflows → Import from File**.
2. Configure the **OpenRouter Chat Model** node with your own OpenRouter API credentials (the exported credential reference is not usable as-is — you'll need to link your own account).
3. Activate the workflow. The webhook path is `/webhook-test/denoising` (`POST`) while testing, or `/webhook/denoising` once the workflow is activated for production use.

### 4. Test it

```bash
curl -X POST "http://localhost:5678/webhook-test/denoising" -F "file=@C:/Users/User/Desktop/jepaproject/agent/test.jpg" -o result.html
```

Open `result.html` in a browser to see the denoised image and metrics.

## Notes

- The OpenRouter model is currently set to `openrouter/free`; swap for a different model in the **OpenRouter Chat Model** node as needed.
- The backend URLs assume a local Docker/n8n setup (`host.docker.internal`) — adjust if deploying elsewhere.
