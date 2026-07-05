# Groq Assistant — backend API + giao diện web
FROM python:3.12-slim

WORKDIR /app

# Cài dependency trước để tận dụng Docker layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY web/ ./web/
COPY api.py chatbot.py ./

# Model embedding cache + database nằm ngoài image (mount volume).
ENV FASTEMBED_CACHE_PATH=/app/model-cache
RUN mkdir -p /app/data /app/model-cache

EXPOSE 8000

# Healthcheck để orchestrator biết app còn sống.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
