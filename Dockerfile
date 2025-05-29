# 1) Base Python image
FROM python:3.11-slim

# 2) System libs for OpenGL / headless CV2 / Torch
RUN apt-get update && \
    apt-get install -y \
      libgl1-mesa-glx \
      libgl1-mesa-dri \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# 3) Copy your code & deps
WORKDIR /app
COPY . /app

# 4) Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5) Expose port & launch Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
