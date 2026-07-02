FROM node:20-alpine

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./frontend/

WORKDIR /app/frontend

RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY frontend ./

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0"]