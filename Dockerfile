FROM python:3.11-slim

# Install OpenGL + CV / headless GUI libs
RUN apt-get update && \
    apt-get install -y \
      libgl1-mesa-glx \
      libgl1-mesa-dri \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
