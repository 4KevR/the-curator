# the-curator

Turning memories into mastery

# Get started

## Prerequisites

- Python 3.10 (client) and 3.12 (server)
- Docker
- Node v22

## Local setup

Due to different dependencies (and their relations), client (CLI application) and server need different Python versions. But both use a shared `pre-commit` setup to enable style consistency in the repository. The configuration is stored in `.pre-commit-config.yaml`. Install the recommended extensions from `.vscode/extensions.json`.

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

Finally, set the defined variables in `.env` and `.env.db` accordingly.

To run LlamaIndex components, start the respective services from the `compose.yml` file:

```bash
docker compose up -d vector-db pgadmin
```

Now, you can run the initial setup - see [here](#terminal).

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

Finally, set the defined variables in `.env.local` accordingly. If you don't know the value of the AUDIO_DEVICE, leave it empty, the CLI will print a list of available devices on startup.

### Client (Web application)

Use Node v22

```bash
npm install
cp src/frontend/.env.local.example src/frontend/.env.local
```

Set the defined variables in `src/frontend/.env.local` accordingly.

## Start

### Terminal

Use startup files directly:

1. `run.py` starts Flask server
2. `run-cli.py` starts CLI application (use defined flags for startup)

For the web client, use `npm run dev` in the `src/frontend` directory.

### VSCode run configurations

Run server and client application via the `Run and Debug` tab

### Docker

You can build and run the server and the web client also in a Docker container. Use `compose.yml` for the setup.

```bash
docker compose up -d --build
```

> [!NOTE]
> Due to loading the LlamaIndex setup, the startup can take some time. See the logs of the container and wait till the setup is completed.

## Additional notes

When using the `LMStudioLLM`, you need to setup 'Meta-Llama-3.1-8B-Instruct' in your local LM Studio installation.
