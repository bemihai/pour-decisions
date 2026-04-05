FROM python:3.11-slim

ENV PYTHONWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get -y install \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install Python dependencies using uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv pip install -r pyproject.toml --system

# copy app files
COPY src/ ./src/
COPY app_config.yml ./

# create cellar-data directory for volumes
RUN mkdir -p /app/cellar-data

EXPOSE 8501
ENV PYTHONPATH=/app

# healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# run streamlit app
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0"]

