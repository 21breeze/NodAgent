FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app


# --------------------------------------------------
# System dependencies
# --------------------------------------------------

RUN apt-get update \
    && apt-get install -y \
        --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*


# --------------------------------------------------
# Python dependencies
# --------------------------------------------------

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --no-cache-dir \
        -r /app/requirements.txt


# --------------------------------------------------
# Application
# --------------------------------------------------

COPY . /app

RUN mkdir -p /app/data/uploads


EXPOSE 8000



CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
