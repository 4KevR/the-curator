# the-curator
Turning memories into mastery

# Get started

## Prerequisites

- Python 3.10
- Docker

## Local setup

```bash
python -m venv .venv
pip install -r requirements-local.txt
cp .env.local.example .env.local
cp .env.example .env
```

Finally, set the defined variables in `.env` and `.env.local` accordingly

## Start

### Terminal

Use startup files directly:

1. `run.py` starts Flask server
2. `run-cli.py` starts CLI application (use defined flags for startup)

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