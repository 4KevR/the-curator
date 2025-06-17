# the-curator
Turning memories into mastery

# Get started

## Prerequisites

- Python 3.10 (client) and 3.12 (server)
- Docker

## Local setup

Due to different dependencies (and their relations), client and server need different Python versions. But both use a shared `pre-commit` setup to enable style consistency in the repository. The configuration is stored in `.pre-commit-config.yaml`. Install the recommended extensions from `.vscode/extensions.json`.

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

### Client

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

## Start

### Terminal

Use startup files directly:

1. `run.py` starts Flask server
2. `run-cli.py` starts CLI application (use defined flags for startup)
3. `run-setup.py` makes setup for LlamaIndex

### VSCode run configurations

Run server and client application via the `Run and Debug` tab

### Docker

You can build and run the Flask server also in a Docker container. Use `compose.yml` for the setup.

```bash
docker compose up -d --build
```

## Additional notes

### Noteworthy:
The 'Meta-Llama-3.1-8B-Instruct' model is deployed locally in LM Studio.
See more details [here](card_generator/create_cards.py).

### To be improved:
pdf_reader: Now it can only be based on text content and cannot interpret images.