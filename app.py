import streamlit as st
import cv2
import pandas as pd
import numpy as np
import re
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
from datetime import datetime
import os
import tempfile

# -----------------------------
# Configuration & models
# -----------------------------
MODEL_PATH = "best.pt"
CHECK_CSV = "check.csv"
GUEST_CSV = "guest.csv"

# load model and OCR
model = YOLO(MODEL_PATH)
ocr = RapidOCR()

# ensure CSV files exist with correct schema
def ensure_csvs():
    if not os.path.exists(CHECK_CSV):
        pd.DataFrame({"plate": []}).to_csv(CHECK_CSV, index=False)
    if not os.path.exists(GUEST_CSV):
        pd.DataFrame({"plate": [], "checkin": [], "checkout": [], "status": []}).to_csv(GUEST_CSV, index=False)

ensure_csvs()

# -----------------------------
# Helpers
# -----------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_check():
    df = pd.read_csv(CHECK_CSV, dtype=str)
    if "plate" not in df.columns:
        df["plate"] = ""
    df["plate"] = df["plate"].fillna("").astype(str).str.upper().str.strip()
    return df

def save_check(df):
    df = df.copy()
    df["plate"] = df["plate"].astype(str).str.upper().str.strip()
    df.to_csv(CHECK_CSV, index=False)

def load_guest():
    # always return dataframe with columns plate, checkin, checkout, status
    if not os.path.exists(GUEST_CSV):
        df = pd.DataFrame({"plate": [], "checkin": [], "checkout": [], "status": []})
        df.to_csv(GUEST_CSV, index=False)
        return df
    df = pd.read_csv(GUEST_CSV, dtype=str)
    for c in ["plate", "checkin", "checkout", "status"]:
        if c not in df.columns:
            df[c] = ""
    df["plate"] = df["plate"].fillna("").astype(str).str.upper().str.strip()
    df["checkin"] = df["checkin"].fillna("").astype(str)
    df["checkout"] = df["checkout"].fillna("").astype(str)
    df["status"] = df["status"].fillna("").astype(str)
    return df

def save_guest(df):
    df = df.copy()
    df["plate"] = df["plate"].astype(str).str.upper().str.strip()
    df["checkin"] = df["checkin"].astype(str)
    df["checkout"] = df["checkout"].astype(str)
    df["status"] = df["status"].astype(str)
    df.to_csv(GUEST_CSV, index=False)

# plate validity: 2 letters start, middle 2-4 alnum, last 4 digits
def is_valid_plate(plate):
    if not plate or not isinstance(plate, str):
        return False
    plate = plate.strip().upper()
    pattern = r"^[A-Z]{2}[A-Z0-9]{2,4}[0-9]{4}$"
    return re.match(pattern, plate) is not None

# preprocess small crop for OCR
def preprocess_plate(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w == 0 or h == 0:
        return None
    target_w, target_h = 220, 80
    if w < target_w:
        fx = target_w / w
        fy = target_h / h
        gray = cv2.resize(gray, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    else:
        gray = cv2.resize(gray, (target_w, target_h))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# rapidocr read + cleanup
def read_plate_rapid(img):
    if img is None:
        return ""
    processed = preprocess_plate(img)
    if processed is None:
        return ""
    try:
        result, _ = ocr(processed)
    except Exception:
        return ""
    if not result:
        return ""
    text = result[0][1]
    text = text.upper().replace(" ", "").replace("-", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    # common confusions
    text = (text.replace("O", "0")
                .replace("I", "1")
                
                )
    return text

# safe YOLO inference -> list of (plate_text, bbox)
def process_frame(frame):
    if frame is None:
        return []
    try:
        results = model.predict(frame, conf=0.5, verbose=False)
    except Exception:
        return []
    plates = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            plate_text = read_plate_rapid(crop)
            plates.append((plate_text, (x1, y1, x2, y2)))
    return plates

# -----------------------------
# Guest / check logic (Option A checkout)
# -----------------------------
def get_status(plate):
    """Return: 'permanent','guest_in','guest_out','not_found'"""
    if not plate:
        return "not_found"
    plate = plate.upper().strip()
    check_df = load_check()
    guest_df = load_guest()
    if plate in check_df["plate"].astype(str).values:
        return "permanent"
    rows = guest_df[guest_df["plate"].astype(str) == plate]
    if not rows.empty:
        last = rows.iloc[-1]
        out = str(last.get("checkout", "")).strip()
        if out == "":
            return "guest_in"
        else:
            return "guest_out"
    return "not_found"

def guest_check_in(plate):
    plate = plate.upper().strip()
    guest_df = load_guest()
    # if last record for that plate has empty checkout => already inside
    rows = guest_df[guest_df["plate"].astype(str) == plate]
    if not rows.empty:
        last = rows.iloc[-1]
        out = str(last.get("checkout", "")).strip()
        if out == "":
            return False
    new = {"plate": plate, "checkin": now_str(), "checkout": "", "status": "IN"}
    guest_df = pd.concat([guest_df, pd.DataFrame([new])], ignore_index=True)
    save_guest(guest_df)
    return True

def guest_check_out(plate):
    plate = plate.upper().strip()
    guest_df = load_guest()
    # find last open record where checkout is empty
    mask = (guest_df["plate"].astype(str) == plate) & (guest_df["checkout"].astype(str).str.strip() == "")
    if mask.any():
        idx = guest_df[mask].index[-1]
        guest_df.at[idx, "checkout"] = now_str()
        guest_df.at[idx, "status"] = "OUT"
        save_guest(guest_df)
        return True
    return False

def add_permanent(plate):
    plate = plate.upper().strip()
    df = load_check()
    if plate in df["plate"].astype(str).values:
        return False
    df = pd.concat([df, pd.DataFrame([{"plate": plate}])], ignore_index=True)
    save_check(df)
    return True

# colored light render
def render_light(status):
    color = "#808080"; text = "Unknown"
    if status == "permanent":
        color = "#16a34a"; text = "Permanent Resident"
    elif status == "guest_in":
        color = "#f59e0b"; text = "Guest (Inside)"
    elif status == "guest_out":
        color = "#0ea5e9"; text = "Guest (Exited)"
    elif status == "not_found":
        color = "#ef4444"; text = "Not Found"
    html = f"""
    <div style="display:flex; align-items:center; gap:12px">
      <div style="width:24px; height:24px; border-radius:50%; background:{color}; box-shadow:0 0 8px {color}"></div>
      <div style="font-weight:600;">{text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart Gate Monitor", layout="wide")
st.title("Smart Society Gate — Number Plate Monitor")

# session for last valid plate
if "last_valid_plate" not in st.session_state:
    st.session_state.last_valid_plate = ""

# layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.header("Input")
    input_type = st.radio("Input Type", ["Image", "Video"], index=1)
    uploaded = st.file_uploader("Upload image/video", type=["jpg","png","jpeg","mp4","mov","avi"])
    st.markdown("---")
    frame_interval = st.slider("Frame skip (video)", 1, 120, 36)
    st.markdown("---")
    manual_plate = st.text_input("Manual plate (optional)", value="")
    if manual_plate:
        manual_plate = manual_plate.upper().strip()
        if st.button("Set Manual Plate"):
            if is_valid_plate(manual_plate):
                st.session_state.last_valid_plate = manual_plate
                st.success("Manual plate set and will be used for actions.")
                st.experimental_rerun()
            else:
                st.error("Manual plate invalid format.")

with col2:
    st.header("Viewer")
    viewer = st.empty()

    if uploaded is not None:
        if input_type == "Image":
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                st.error("Unable to read image.")
            else:
                plates = process_frame(frame)
                drawn = frame.copy()
                for plate, (x1, y1, x2, y2) in plates:
                    if is_valid_plate(plate):
                        st.session_state.last_valid_plate = plate
                    status = get_status(plate) if plate else "not_found"
                    if status == "permanent":
                        color = (0, 200, 0)
                    elif status == "guest_in":
                        color = (0, 180, 255)
                    elif status == "guest_out":
                        color = (255, 180, 0)
                    elif is_valid_plate(plate):
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 255)
                    cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(drawn, plate, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                viewer.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        else:
            # video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            tfile.flush()
            cap = cv2.VideoCapture(tfile.name)
            frame_no = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_no += 1
                if frame_no % frame_interval == 0:
                    plates = process_frame(frame)
                    for plate, (x1, y1, x2, y2) in plates:
                        if is_valid_plate(plate):
                            st.session_state.last_valid_plate = plate
                        status = get_status(plate) if plate else "not_found"
                        if status == "permanent":
                            color = (0, 200, 0)
                        elif status == "guest_in":
                            color = (0, 180, 255)
                        elif status == "guest_out":
                            color = (255, 180, 0)
                        elif is_valid_plate(plate):
                            color = (0,0,255)
                        else:
                            color = (0,255,255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, plate, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                viewer.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            try:
                os.unlink(tfile.name)
            except Exception:
                pass

with col3:
    st.header("Result & Actions")

    if st.session_state.last_valid_plate:
        st.write("Latest valid-format plate detected:")
        st.code(st.session_state.last_valid_plate)
    else:
        st.write("No valid-format plate detected yet.")

    edited_plate = st.text_input("Edit plate before action:", value=st.session_state.last_valid_plate).upper().strip()

    status = get_status(edited_plate) if edited_plate else "not_found"
    render_light(status)
    st.markdown("---")

    # Actions (only active if edited_plate is valid format)
    if edited_plate and is_valid_plate(edited_plate):
        if status == "permanent":
            st.success("Permanent resident recognized.")
            if st.button("Force Guest Check-In (temporary)"):
                ok = guest_check_in(edited_plate)
                if ok:
                    st.success(f"Guest check-in recorded for {edited_plate} at {now_str()}")
                else:
                    st.warning("Already checked in as guest.")
                st.session_state.last_valid_plate = ""
                st.experimental_rerun()

        elif status == "guest_in":
            st.warning("Guest is currently inside.")
            if st.button("Check-Out Guest"):
                ok = guest_check_out(edited_plate)
                if ok:
                    st.success(f"Guest checked out: {edited_plate} at {now_str()}")
                else:
                    st.error("Could not check out (no open guest record).")
                st.session_state.last_valid_plate = ""
                st.experimental_rerun()

        elif status == "guest_out":
            st.info("Guest previously visited and already checked out.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Check-In Guest Again"):
                    ok = guest_check_in(edited_plate)
                    if ok:
                        st.success(f"Guest checked in: {edited_plate} at {now_str()}")
                    else:
                        st.warning("Already checked in.")
                    st.session_state.last_valid_plate = ""
                    st.experimental_rerun()
            with c2:
                if st.button("Add to Permanent (check.csv)"):
                    ok = add_permanent(edited_plate)
                    if ok:
                        st.success(f"{edited_plate} added to permanent residents.")
                    else:
                        st.info("Plate already in permanent list.")
                    st.session_state.last_valid_plate = ""
                    st.experimental_rerun()

        else:  # not_found
            st.error("Plate not found in check.csv or guest.csv.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Add as Guest Check-In"):
                    ok = guest_check_in(edited_plate)
                    if ok:
                        st.success(f"Guest check-in recorded for {edited_plate} at {now_str()}")
                    else:
                        st.warning("Already checked in as guest.")
                    st.session_state.last_valid_plate = ""
                    st.experimental_rerun()
            with c2:
                if st.button("Add to Permanent (check.csv)"):
                    ok = add_permanent(edited_plate)
                    if ok:
                        st.success(f"{edited_plate} added to permanent residents.")
                    else:
                        st.info("Plate already in permanent list.")
                    st.session_state.last_valid_plate = ""
                    st.experimental_rerun()
    else:
        if edited_plate and not is_valid_plate(edited_plate):
            st.error("Edited plate not valid (2 letters start, 2-4 middle alnum, 4 digits end).")
        st.info("Edit detected plate or type one manually to take action.")

st.markdown("---")
if st.checkbox("Show CSVs (debug)"):
    st.write("Permanent residents (check.csv):")
    st.dataframe(load_check())
    st.write("Guest log (guest.csv):")
    st.dataframe(load_guest())
