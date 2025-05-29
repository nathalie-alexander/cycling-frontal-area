# Use a slim Python base image
FROM python:3.11-slim

# 1) Install system dependencies
RUN apt-get update && \
    apt-get install -y \
      libgl1-mesa-glx \
      libgl1-mesa-dri \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# 2) Copy your app into the container
WORKDIR /app
COPY . /app

# 3) Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4) Launch Streamlit on port 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
