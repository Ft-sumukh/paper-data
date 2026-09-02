FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API and Streamlit dashboard ports
EXPOSE 8000
EXPOSE 8501

# Default entrypoint starts the FastAPI service
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
