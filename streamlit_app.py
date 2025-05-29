# app.py
import streamlit as st
import numpy as np
from ultralytics import YOLO

st.title("Test App")
model = YOLO("yolov8n-seg.pt")
st.success("Model loaded successfully")
