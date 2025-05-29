import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
import plotly.express as px

# ————————————————
# 1. Load model & set up
# ————————————————
@st.cache_resource
def load_model():
    return YOLO("yolov8m-seg.pt")

def load_csv(uploader):
    # rewind to the start of the buffer
    uploader.seek(0)
    try:
        df = pd.read_csv(uploader)
    except pd.errors.EmptyDataError:
        st.error("The CSV you uploaded is empty or couldn’t be parsed.")
        st.stop()
    return df

model = load_model()

st.title("🎥 Person Segmentation from Video")
st.markdown(
    """
    ## Why Measure Frontal Area in Cycling?  
    Aerodynamic drag is the single largest resistive force a cyclist must overcome above ~15 km/h.  
    Because drag grows with the square of velocity, even small reductions in the rider–bicycle frontal area  
    can yield significant power savings.  

    **Drag‐force equation**  
    $$
    F_R \;=\; \frac{c_w \,\cdot\, A \,\cdot\, \rho \,\cdot\, v^2}{2}
    $$
    where  
    - $F_R$ is the resistive (drag) force  
    - $c_w$ is the drag coefficient  
    - $A$ is the frontal area  
    - $\rho$ is the air density  
    - $v$ is the relative wind speed  

    **Note:** extracting the silhouette from a frontal video gives only a **rough estimate** of the true frontal area and possible savings related to position changes. 
    """
)



st.markdown(
    """
    **How to use:**  
    1. **Upload a video** filmed from a straight-on (frontal) perspective so that the person’s silhouette is fully visible and you capture the true frontal area.  
    2. The video file should contain all the different positions you want to analyse (no separate files). 
    3. **Ensure good contrast** between subject and background (e.g. plain wall or uniform backdrop) to improve mask accuracy.  
    4. **Keep clips short** (30–60 s) to limit processing time.  
    5. **Optionally**, if you’ve already run a segmentation before, upload your CSV to skip processing and immediately re-plot your results (don't forget to also upload the corresponding video).  
    """
)


# ————————————————
# Uploaders
# ————————————————
col1, col2 = st.columns(2)
uploaded_video = col1.file_uploader("📁 Video file", type=["mp4", "mov"])
uploaded_csv   = col2.file_uploader("📁 CSV with frame & area", type=["csv"])

# ————————————————
# Session-state initialization
# ————————————————
for key, default in [("processing", False), ("stop_processing", False),
                     ("frame_numbers", []), ("person_areas", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

video_path = None
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    tmp_cap = cv2.VideoCapture(video_path)
    fps = tmp_cap.get(cv2.CAP_PROP_FPS)
    tmp_cap.release()

if uploaded_csv is not None and video_path is None:
    df = load_csv(uploaded_csv)
    if {"frame","person_area"}.issubset(df.columns):
        fig = px.line(
            df, x="frame", y="person_area",
            title="Frontal area over time",
            labels={"frame":"Frame","person_area":"Person Area"}
        )
        st.plotly_chart(fig)
    else:
        st.error("CSV must contain `frame` and `person_area` columns.")
    st.stop()  # done for CSV-only

# ————————————————
# Video path: show controls & slider once
# ————————————————
if uploaded_video is None:
    st.info("Choose a video to start segmentation.")
    st.stop()

if uploaded_csv is None:
    st.write(f"Video: {fps:.0f} frames per second")

    max_skip     = max(1, int(fps // 2))
    default_skip = max(1, int(fps // 6))
    frame_skip   = st.slider(
        "Frame skip (accuracy ↔ speed)",
        1, max_skip, default_skip, 1
    )

    # Start / Stop buttons
    col1, col2 = st.columns(2)
    start_clicked = col1.button("▶️ Start Processing")
    stop_clicked  = col2.button("⏹️ Stop")

    # Kick off processing only on Start click
    if start_clicked:
        st.session_state.processing      = True
        st.session_state.stop_processing = False
        # reset old data
        st.session_state.frame_numbers = []
        st.session_state.person_areas  = []

    if stop_clicked:
        st.session_state.stop_processing = True

    # Placeholders
    col1, col2 = st.columns(2)
    img_pl   = col1.empty()
    chart_pl = col2.empty()

    # ————————————————
    # Processing loop
    # ————————————————
    if st.session_state.processing and not st.session_state.stop_processing:
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success or st.session_state.stop_processing:
                break

            if idx % frame_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                H, W = frame.shape[:2]

                # inference
                res = model(rgb)[0]
                mask = np.zeros((H, W), dtype=bool)
                if res.masks:
                    for m, cls in zip(res.masks.data.cpu().numpy(),
                                    res.boxes.cls.cpu().numpy().astype(int)):
                        if model.names[int(cls)] == "person":
                            m_resized = cv2.resize(m, (W, H)) > 0.5
                            mask |= m_resized

                # overlay & display
                overlay = rgb.copy()
                overlay[mask] = (overlay[mask] * 0.5 + np.array([0,255,0])*0.5).astype(np.uint8)
                img_pl.image(overlay, caption=f"Frame {idx}")

                # store & chart
                st.session_state.person_areas.append(mask.sum())
                st.session_state.frame_numbers.append(idx)
                chart_pl.line_chart({
                    "Person Area": st.session_state.person_areas
                })

            idx += 1

        cap.release()
        # final message
        if st.session_state.stop_processing:
            st.warning("⏹️ Processing stopped by user.")
        else:
            st.success("✅ Processing complete!")
        st.session_state.processing = False  # prevent rerun

# ————————————————
# Export button (always shown if we have data)
# ————————————————
if st.session_state.person_areas:
    df = pd.DataFrame({
        "frame": st.session_state.frame_numbers,
        "person_area": st.session_state.person_areas
    })
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Export CSV",
        data=csv,
        file_name="person_area_over_time.csv",
        mime="text/csv"
    )

# ————————————————
# CSV  VIDEO → IMMEDIATELY LOAD CSV INTO SESSION
# ————————————————
if uploaded_csv is not None and video_path is not None:
    df = load_csv(uploaded_csv)
    if not {"frame","person_area"}.issubset(df.columns):
        st.error("CSV needs `frame` & `person_area` columns.")
        st.stop()

    # overwrite session-state lists from CSV
    st.session_state.frame_numbers = df["frame"].tolist()
    st.session_state.person_areas  = df["person_area"].tolist()

analysis_ready = video_path is not None and len(st.session_state.frame_numbers) > 0
if analysis_ready:
    if uploaded_csv:
        df = load_csv(uploaded_csv)
    else:
        df = pd.DataFrame({
            "frame": st.session_state.frame_numbers,
            "person_area": st.session_state.person_areas
        })
    if not {"frame","person_area"}.issubset(df.columns):
        st.error("CSV needs `frame` & `person_area` columns.")
        st.stop()

    col_frame, col_regions = st.columns(2)

    # 1) pick your current frame
    min_f = min(st.session_state.frame_numbers)
    max_f = max(st.session_state.frame_numbers)
    current_frame = col_frame.slider(
        "🔢 Select frame to inspect the video",
        min_value=min_f, max_value=max_f, value=min_f, step=1
    )

    # pick your regions (baseline + extras)
    n_pos = col_regions.slider(
        "Number of positions in addition to baseline",
        min_value=1, max_value=4, value=2, step=1
    )
    cols = st.columns(n_pos + 1)
    regions = {}
    for i, col in enumerate(cols):
        label = "Baseline" if i == 0 else f"Position {i}"
        default_start = min_f + i * ((max_f - min_f) // (n_pos + 1))
        default_end   = default_start + ((max_f - min_f) // (n_pos + 1))
        regions[label] = col.slider(
            f"{label} range",
            min_value=min_f,
            max_value=max_f,
            value=(default_start, default_end),
            step=5
        )

    # build ONE figure
    fig = px.line(
        df, x="frame", y="person_area",
        title="Frontal area over time",
        labels={"frame":"Frame","person_area":"Person Area"}
    )

    # vertical marker for current frame
    fig.add_vline(
        x=current_frame,
        line_dash="dash",
        annotation_text="▶",
        annotation_position="top"
    )

    # translucent rectangles for each region
    palette = px.colors.qualitative.Plotly
    for idx, (label, (x0, x1)) in enumerate(regions.items()):
        color = palette[idx % len(palette)]
        fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=x0, x1=x1,
            y0=0, y1=1,
            fillcolor=color,
            opacity=0.2,
            layer="below",
            line_width=0
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=1.02,
            xref="x", yref="paper",
            text=label,
            showarrow=False,
            font=dict(color=color)
        )

    col_img, col_chart = st.columns([1, 2])
    col_chart.plotly_chart(fig)

    # grab & display the video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame))
    ok, frame = cap.read()
    cap.release()
    if ok:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        col_img.image(rgb, caption=f"Frame {current_frame}")
    else:
        st.error(f"Could not read frame {current_frame}.")

    # extract the numeric data per region
    frames = st.session_state.frame_numbers
    areas  = st.session_state.person_areas
    region_data = {
        label: [
            area for f, area in zip(frames, areas)
            if start <= f <= end
        ]
        for label, (start, end) in regions.items()
    }
    
    baseline_vals = region_data["Baseline"]
    baseline_mean = np.mean(baseline_vals) if baseline_vals else 0

    pct_diffs = {}
    for label, vals in region_data.items():
        if label == "Baseline": 
            continue
        mean_val = np.mean(vals) if vals else 0
        pct = (mean_val - baseline_mean) / baseline_mean * 100 if baseline_mean else 0
        pct_diffs[label] = pct

    if pct_diffs:
        for i, (label, pct) in enumerate(pct_diffs.items(), start=1):
            col = cols[i]  # the slider for Position i lives in cols[i]
            color = "red" if pct >= 0 else "green"
            sign  = "+" if pct >= 0 else ""
            # col.markdown(
            #     f"""
            #     <div style="text-align: center;">
            #     <p style='margin: 0; font-weight: bold;'>{label}</p>
            #     <h2 style='margin: 0; font-size: 2.0em;'>
            #         <span style='color: {color};'>{sign}{pct:.0f}%</span>
            #     </h2>
            #     </div>
            #     """,
            #     unsafe_allow_html=True
            # )
            col.markdown(
                f"""
                <div style="text-align: right;">
                    <p style="margin: 0; font-size: 1.5em;">
                        {label}: <span style="color: {color}; font-weight: bold;">{sign}{pct:.0f}%</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write("No additional regions selected.")

