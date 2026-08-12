FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY notebooks/ notebooks/

EXPOSE 8000 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.headless=true"]
