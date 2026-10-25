# Insurance Support Bot API (Experiment 7)

Services: LM Studio (model server, :8317) -> FastAPI (:8000) -> Chainlit UI (:8001). MLflow UI (:5000) is optional.
Run every command from the folder named in the step. `.env` files and `mlflow.db` are relative to that folder.

## 1. LM Studio
Load your model, set Context Length to at least 4096, open the Developer tab and start the server.
Copy the model identifier shown there into `LLM_MODEL` in `api/.env`.
Copy the server address (e.g. `http://127.0.0.1:8317/v1`) into `LLM_BASE_URL` in `api/.env`.

## 2. API (folder: api/)
    uv venv .venv
    .venv\Scripts\activate             # Windows
    uv pip install -r requirements.txt
    cp .env.example .env               # then fill in LLM_MODEL, LLM_BASE_URL, QDRANT_URL, QDRANT_API_KEY
    uv run python -m scripts.ingest    # once; rebuilds the Qdrant collection and writes data/taxonomy.json
    uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

Swagger UI: http://127.0.0.1:8000/docs

## 3. Chat UI (folder: ui/, separate virtual environment)
    uv venv .venv
    .venv\Scripts\activate             # Windows
    uv pip install -r requirements.txt
    cp .env.example .env
    uv run chainlit run chat_ui.py --port 8001

## 4. MLflow UI (folder: api/, optional)
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

## 5. Lab tests (folder: api/, API must be running)
    uv run python -m scripts.run_tests                       # includes the manual "stop LM Studio" test
    uv run python -m scripts.run_tests --skip-backend-down   # skips it
Output: observation_table_lab.csv and observation_table_rag.csv

## Notes on parallel requests
LM Studio's "Parallel" setting (default 4) is matched by `LLM_PARALLEL=4` in `api/.env`.
The API uses a `threading.Semaphore(LLM_PARALLEL)` so up to 4 graph runs execute concurrently,
each making independent LLM calls. Requests beyond the limit wait up to `QUEUE_TIMEOUT_S` seconds.
