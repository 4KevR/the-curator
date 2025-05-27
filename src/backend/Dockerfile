FROM python:3.13-slim AS base

WORKDIR /flask-app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM base

COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 5000

ENTRYPOINT ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.backend.application:app"]
