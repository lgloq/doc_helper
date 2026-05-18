FROM node:20-slim

WORKDIR /app/frontend

COPY frontend/package.json /app/frontend/package.json
RUN npm install

COPY frontend /app/frontend

CMD ["sh", "-lc", "npm run dev -- --host 0.0.0.0 --port ${FRONTEND_PORT:-18073}"]
