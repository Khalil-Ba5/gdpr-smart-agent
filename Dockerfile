FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and scraped data. The Chroma index (chroma_db/) is large and
# rebuildable, so it is mounted as a volume at run time rather than baked in.
COPY src ./src
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
