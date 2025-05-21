FROM python:3.13-alpine as base

WORKDIR /flask-app

RUN apk add --no-cache \
    python3-dev \
    py3-setuptools \
    build-base \
    libffi-dev \
    openssl-dev \
    gcc \
    musl-dev \
    linux-headers

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM base

COPY . .

ENTRYPOINT ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.backend.application:app"]
