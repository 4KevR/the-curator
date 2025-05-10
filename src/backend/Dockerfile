FROM python:3.11-slim as base

WORKDIR /flask-app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends python3-dev  \
    && apt-get install -y --no-install-recommends python3-setuptools  \
    && apt-get install -y --no-install-recommends build-essential libssl-dev libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM base

COPY . .

ENTRYPOINT ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.backend.application:app"]
