FROM node:20-slim

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=7860
ENV AGENTARENA_HOSTED=1

COPY . /app

EXPOSE 7860

CMD ["node", "server.js"]
