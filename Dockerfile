# Use official Python image
FROM python:3.12-slim-bookworm@sha256:secure-digest-value as base

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*


# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install timesfm from source (replace <REPO_LINK_HERE> with actual repo)
RUN git clone <REPO_LINK_HERE> && cd timesfm && pip install -e .

# Copy all project files
COPY . .

# --- API Image ---
FROM base as api
EXPOSE 8000
CMD [".venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Dashboard Image ---
FROM base as dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
