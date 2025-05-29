# app.py
import types, sys
sys.modules['torch.classes'] = types.ModuleType("torch.classes")

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.title("Test App")
model = YOLO("yolov8n-seg.pt")
st.success("Model loaded successfully")
