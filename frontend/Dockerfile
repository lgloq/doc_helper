# syntax=docker/dockerfile:1.7
FROM node:20-slim

WORKDIR /app/frontend

COPY frontend/package*.json /app/frontend/
RUN --mount=type=cache,target=/root/.npm \
    if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY frontend /app/frontend

CMD ["sh", "-lc", "npm run dev -- --host 0.0.0.0 --port ${FRONTEND_PORT:-18073}"]
