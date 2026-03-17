FROM python:3.11-slim

WORKDIR /app

ENV PIP_REQUIRE_HASHES=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=200

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt

COPY app.py .

EXPOSE 9011

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9011"]
