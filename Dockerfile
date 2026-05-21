# API-only image: FastAPI + retrieval (sentence-transformers).
# Streamlit stays local; see README "Docker (FastAPI)".

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BELGRADE_RECOMMENDER_ROOT=/app

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY fixtures ./fixtures

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[api,retrieve]"

EXPOSE 8080

CMD ["uvicorn", "belgrade_recommender.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
