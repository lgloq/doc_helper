# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app/backend

COPY backend/pyproject.toml /app/backend/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && python -c "import tomllib; data = tomllib.load(open('pyproject.toml', 'rb')); open('/tmp/requirements.txt', 'w', encoding='utf-8').write('\n'.join(data['project']['dependencies']) + '\n')" \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY backend /app/backend

RUN pip install --no-deps -e .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
