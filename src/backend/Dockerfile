FROM python:3.12-slim AS base

WORKDIR /flask-app

COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt
RUN pip install --no-cache-dir gevent

FROM base

COPY . .

RUN useradd -m appuser

# Ensure data directory exists and is owned by appuser
RUN mkdir -p ./data && chown -R appuser:appuser ./data
RUN mkdir -p ./model_cache && chown -R appuser:appuser ./model_cache

USER appuser

# Download NLTK models before starting the app
RUN python -c "import nltk; nltk.download('punkt_tab')"

EXPOSE 5000

ENTRYPOINT ["gunicorn", "-w", "1", "-k", "gevent", "-b", "0.0.0.0:5000", "--log-level=info", "--timeout", "0", "src.backend.application:app"]
