# Build from the src/ directory:  docker build -f deploy/Dockerfile -t audiodf .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY audiodf ./audiodf
COPY configs ./configs

# Model artifacts are mounted (volume / PVC), not baked into the image.
ENV AUDIODF_ARTIFACTS=/models PYTHONUNBUFFERED=1 PYTHONPATH=/app
EXPOSE 8000 9100

CMD ["uvicorn", "audiodf.serving.api:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
