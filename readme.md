# 🚗 Smart Society Gate – Number Plate Monitoring System

A **Streamlit-based smart gate monitoring system** that automatically detects vehicle number plates using **YOLO + OCR**, identifies **permanent residents vs guests**, and manages **check-in / check-out logs** in real time.

This system is suitable for **gated societies, campuses, offices, and parking areas**.

---

## ✨ Features

- 📸 **Image & Video Input**
  - Upload images or videos for number plate detection
  - Frame skipping supported for faster video processing

- 🔍 **Automatic Number Plate Detection**
  - YOLO model for plate localization
  - RapidOCR (ONNX) for text recognition
  - Plate format validation

- 🏠 **Resident Classification**
  - **Permanent Residents** (stored in `check.csv`)
  - **Guests** with entry & exit tracking (`guest.csv`)

- 🟢 **Smart Status Indicator**
  - Green → Permanent Resident  
  - Yellow → Guest (Inside)  
  - Blue → Guest (Exited)  
  - Red → Not Found  

- ✍️ **Manual Plate Override**
  - Edit or manually enter plate numbers before actions

- 📊 **CSV-Based Logging**
  - Simple, lightweight, and easy to audit
  - No external database required

---

## Plate Format Validation

Plates must follow this format:

    AA00AA0000

Rules:
- Starts with 2 letters
- Middle 2–4 alphanumeric characters
- Ends with 4 digits

Examples:
    MH12AB1234
    DL8CAF5030


--------------------------------------------------

## Project Structure

    .
    ├── app.py                # Main Streamlit application
    ├── best.pt               # YOLO trained model (number plate detection)
    ├── check.csv             # Permanent resident plates
    ├── guest.csv             # Guest entry/exit logs
    ├── requirements.txt
    └── README.md


--------------------------------------------------

## Installation

Step 1: Clone the Repository

    git clone https://github.com/your-username/smart-gate-monitor.git
    cd smart-gate-monitor


Step 2: Create Virtual Environment (Recommended)

    python -m venv venv
    source venv/bin/activate      # Linux / Mac
    venv\Scripts\activate         # Windows


Step 3: Install Dependencies

    pip install -r requirements.txt


--------------------------------------------------

## Running the App

    streamlit run app.py

Open browser at:

    http://localhost:8501


--------------------------------------------------

## Input Options

Image formats:
- JPG
- PNG
- JPEG

Video formats:
- MP4
- MOV
- AVI

You can:
- Upload a file
- Skip frames in videos
- Manually correct plate text before actions


--------------------------------------------------

## CSV Files Explained

check.csv (Permanent Residents)

    plate
    MH12AB1234
    DL8CAF5030


guest.csv (Guest Log)

    plate,checkin,checkout,status
    MH14XY9999,2026-01-31 10:20:10,,IN
    MH14XY9999,2026-01-31 18:45:02,2026-01-31 22:10:44,OUT


--------------------------------------------------

## Tech Stack

- Frontend: Streamlit
- Detection: YOLO (Ultralytics)
- OCR: RapidOCR (ONNX Runtime)
- Computer Vision: OpenCV
- Data Handling: Pandas, NumPy


--------------------------------------------------

## Future Improvements

- Role-based access (Guard / Admin)
- Cloud database (PostgreSQL / Firebase)
- Live CCTV stream support
- Barrier / Gate automation
- Analytics dashboard (daily / monthly traffic)


--------------------------------------------------

## Author

Anisha Kumari Lal 
M.Tech - Computer Science and Engineering 
NIT Allahabad

Raj Jagannath Nangare  
M.Tech – Information Security  
NIT Allahabad  


--------------------------------------------------

## License

This project is for educational and research purposes.  
You are free to modify and extend it.
