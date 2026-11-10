FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src:/app

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY . .

RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 astrobot \
    && chown -R astrobot:astrobot /app

USER astrobot

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "astrobot.web:app", "--host", "0.0.0.0", "--port", "8080"]
