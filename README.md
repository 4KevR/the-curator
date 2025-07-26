# the-curator

Turning memories into mastery

# Get started

## Prerequisites

- Python 3.10 (client) and 3.12 (server)
- Docker
- Node v22

## Local setup (development)

Due to different dependencies (and their relations), client (only CLI application) and server need different Python versions. But both use a shared `pre-commit` setup to enable style consistency in the repository. The configuration is stored in `.pre-commit-config.yaml`. Install the recommended extensions from `.vscode/extensions.json`.

### Server

Use Python 3.12

```bash
python -m venv .venv.server
# Activate envionment based on system (Mac: source .venv.server/bin/activate)
pip install -r requirements-server.txt
pip install -r requirements-local.txt
cp .env.db.example .env.db
cp .env.example .env
pre-commit install
```

Finally, set the defined variables in `.env` and `.env.db` accordingly. For an example of the `.env`, see [here](#development-setup).

To run LlamaIndex components, start the respective services from the `compose.yml` file:

```bash
docker compose up -d vector-db pgadmin
```

### Client (CLI application)

Use Python 3.10

```bash
python -m venv .venv
# Activate envionment based on system (Mac: source .venv/bin/activate)
pip install -r requirements-client.txt
pip install -r requirements-local.txt
cp .env.local.example .env.local
pre-commit install
```

Finally, set the defined variable in `.env.local` accordingly. If you don't know the value of the AUDIO_DEVICE, leave it empty, the CLI will print a list of available devices on startup.

### Client (Web application)

Use Node v22

```bash
cd src/frontend
npm install
cp .env.local.example .env.local
```

Set the defined variable in `src/frontend/.env.local` accordingly. For an example, see [here](#development-setup).

## Start

> [!NOTE]
> Due to loading the LlamaIndex setup, the startup can take some time. See the logs and wait until the setup is completed.

### Terminal

> [!IMPORTANT]
> Requires setup for development described [here](#local-setup-development)

Use startup files directly:

1. `run.py` starts Flask server
2. `run-cli.py` starts CLI application (use defined flags for startup)

For the web client, use `npm run dev` in the `src/frontend` directory.

### VSCode run configurations

> [!IMPORTANT]
> Requires setup for development described [here](#local-setup-development)

Run server and client application via the `Run and Debug` tab

### Docker

You can alternatively build and run the server and the web client also in a Docker container. Use `compose.yml` for the setup. For an example of the `.env` and `src/frontend/.env.local` file, see [here](#docker-setup).

```bash
docker compose up -d --build
```

## Access

Development setup: [http://127.0.0.1:3050](http://127.0.0.1:3050)

Docker: [http://127.0.0.1:7070](http://127.0.0.1:7070)

## Additional notes

When using the `LMStudioLLM`, you need to setup 'Meta-Llama-3.1-8B-Instruct' in your local LM Studio installation.

### Examples for .env files

#### Development setup

##### .env

```bash
LECTURE_TRANSLATOR_TOKEN="<TOKEN>"
LECTURE_TRANSLATOR_URL="<URL>"
LLM_URL="<URL>"
ANKI_COLLECTION_PATH="./data/anki_env"
HUGGING_FACE_TOKEN="<TOKEN>"
FRONTEND_URL="http://127.0.0.1:3050"
ANKI_RELATIVE_DATA_DIR="./data/anki_files"
LLM_TO_USE="hosted"  # use local when LM Studio is running, "hosted" will use the defined "LLM_URL" alongside a HuggingFace InferenceClient
```

##### src/frontend/.env.local

```bash
NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:5000"
```

#### Docker setup

##### .env

```bash
LECTURE_TRANSLATOR_TOKEN="<TOKEN>"
LECTURE_TRANSLATOR_URL="<URL>"
LLM_URL="<URL>"
ANKI_COLLECTION_PATH="/flask-app/data/anki_env"
HUGGING_FACE_TOKEN="<TOKEN>"
FRONTEND_URL="http://127.0.0.1:7070"
ANKI_RELATIVE_DATA_DIR="/flask-app/data/anki_files"
LLM_TO_USE="hosted"
```

##### src/frontend/.env

```bash
NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:7071"
```
