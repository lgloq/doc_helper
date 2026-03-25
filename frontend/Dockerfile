FROM node:20-slim

WORKDIR /app/frontend

COPY frontend/package.json /app/frontend/package.json
RUN npm install

COPY frontend /app/frontend

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
