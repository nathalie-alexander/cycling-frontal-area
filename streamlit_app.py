import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt

# Load the model (lightweight version)
@st.cache_resource
def load_model():
    return YOLO("yolov8m-seg.pt")

model = load_model()

st.title("🎥 Person Segmentation from Video")
st.markdown("Uploads a video and tracks the 'person' mask over time.")

uploaded_video = st.file_uploader("📁 Upload a video file", type=["mp4", "mov"])

# --- Initialize session state ---
if "processing" not in st.session_state:
    st.session_state.processing = False

if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False

# --- Control buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Processing"):
        st.session_state.processing = True
        st.session_state.stop_processing = False

with col2:
    if st.button("⏹️ Stop"):
        st.session_state.stop_processing = True

processing_status = 0

if uploaded_video:
    # Save to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.write(f"Video FPS: {fps:.2f}")

    frame_skip = st.slider("Frame skip (speed vs. accuracy)", 1, 10, 3)

    frame_idx = 0
    person_areas = []
    frame_numbers = []
    
    col1, col2 = st.columns(2)
    img_placeholder = col1.empty()
    chart_placeholder = col2.empty()

    while cap.isOpened() and st.session_state.processing and not st.session_state.stop_processing:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]
            person_mask = np.zeros((H, W), dtype=bool)

            # Run inference
            results = model(frame_rgb)
            r = results[0]

            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[i] for i in class_ids]

                for mask, cls_name in zip(masks, class_names):
                    if cls_name == "person":
                        resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                        person_mask = np.logical_or(person_mask, resized)

            # Overlay mask
            overlay = frame_rgb.copy()
            overlay[person_mask] = (
                0.5 * overlay[person_mask] + 0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)

            # Update display
            img_placeholder.image(overlay, channels="RGB", caption=f"Frame {frame_idx}")

            person_areas.append(person_mask.sum())
            frame_numbers.append(frame_idx)

            # Update plot
            chart_placeholder.line_chart({
                # "Frame": frame_numbers,
                "Person Area": person_areas
            })

        frame_idx += 1

    cap.release()
    processing_status = 1
    st.success("✅ Video processing complete!")

if processing_status == 1:

    # Show final static chart
    st.subheader("📈 Final Person Mask Area Over Time")
    