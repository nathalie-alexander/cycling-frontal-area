import streamlit as st
#import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

st.title("🎈 My new app")
st.write(
    "Let's start building!"
)

# Load model (small or nano for speed)
#model = YOLO("yolov8n-seg.pt")

st.title("Live Video Mask Dashboard")
st.markdown("Shows 'person' mask overlay and area plot")

# Upload video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov"])
if uploaded_video:
    # Save to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    # Initialize chart + image slot
    img_slot = st.empty()
    chart_slot = st.line_chart([], height=200)
    person_areas = []

    # Frame skip
    skip = st.slider("Frame skip (higher = faster)", 1, 10, 3)

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]
            person_mask = np.zeros((H, W), dtype=bool)

            # Run YOLO
            results = model(frame_rgb)
            r = results[0]

            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[i] for i in class_ids]

                for mask, cls in zip(masks, class_names):
                    if cls == "person":
                        resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                        person_mask = np.logical_or(person_mask, resized)

            # Apply green overlay
            overlay = frame_rgb.copy()
            overlay[person_mask] = (
                0.5 * overlay[person_mask] + 0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)

            # Update dashboard
            img_slot.image(overlay, channels="RGB", caption=f"Frame {frame_idx}")
            person_areas.append(person_mask.sum())
            chart_slot.add_rows([person_areas[-1]])

        frame_idx += 1

    cap.release()
