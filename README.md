# real-time-road-anomaly-raspberrypi
Real-time road anomaly detection on Raspberry Pi 4 using YOLO11n (ONNX Runtime). Detects potholes (with diameter estimation), vehicles (motion classification), animals, and obstacles from recorded dashcam footage at ~5 FPS.

# 🚗 Real-Time Road Anomaly Detection on Raspberry Pi 4  
### Bharat AI SoC Student Challenge – Problem Statement 3

---

## 📌 Project Overview

This project implements a real-time Edge AI system on Raspberry Pi 4 for detecting road anomalies from recorded dashcam footage.

The system detects:

- Potholes (with diameter estimation in pixels)
- Vehicles (with motion classification: Moving / Stationary / Unknown)
- Animals

The model is based on YOLO11n, exported to ONNX format, and deployed using ONNX Runtime on Raspberry Pi.

All detections are logged in a structured CSV file, and the annotated output video displays FPS and frame statistics.

---

## 🎯 Objective

- Achieve ≥5 FPS inference on Raspberry Pi 4  
- Detect and log road anomalies from recorded video  
- Maintain high precision with reduced false positives  
- Perform complete edge processing (no cloud dependency)

---

## 🛠 Hardware Used

- Raspberry Pi 4  
- 64-bit Raspberry Pi OS  
- High-speed microSD card  
- Input Source: Recorded MP4 dashcam footage  

---

## 💻 Software Stack

- Python 3.9+  
- OpenCV  
- NumPy  
- Pandas  
- ONNX Runtime  
- YOLO11n (custom-trained for pothole detection)

---

## 🧠 Model Details

- Base Model: YOLO11n  
- Custom-trained for pothole detection  
- Exported Model: `best.onnx`  
- Inference Engine: ONNX Runtime  
- Deployment Mode: CPU (ARM Cortex-A72)  
- Average Performance: ~5 FPS  

---

## ⚙️ System Architecture
Recorded Dashcam Video (MP4)
            ↓
OpenCV Video Capture
            ↓
Frame Preprocessing
(Resize → Normalize → Format Conversion)
            ↓
ONNX Runtime Inference (YOLO11n - ARM CPU)
            ↓
Post-Processing
(NMS + Confidence Filtering)
            ↓
Object Classification
(Potholes / Vehicles / Animals / Obstacles)
            ↓
Feature Extraction
- Diameter Estimation (Potholes)
- Motion Classification (Vehicles)
            ↓
Logging Module
(CSV File Storage)
            ↓
Output Display
(Bounding Boxes + FPS + Frame Count)







---

## 📊 Output Logging

All detections are saved in a structured CSV file (`detection_log.csv`).

### CSV Format

| Serial_Number | Frame_Number | Class | Confidence | BBox (x,y,w,h) | Diameter | Motion_Status |
|--------------|-------------|--------|------------|----------------|----------|---------------|

- **Class** → pothole / vehicle / animal  
- **Diameter** → Calculated for potholes (in pixels)  
- **Motion_Status** → Moving / Stationary / Unknown (for vehicles)  

---

## 📂 Repository Structure
real-time-road-anomaly-raspberrypi/
│
├── README.md # Project documentation
├── requirements.txt # Python dependencies
│
├── models/ # Trained and exported models
│ └── best.onnx # YOLO11n ONNX model
│
├── src/ # Source code
│ └── main.py # Main inference pipeline
│
├── data/ # Input and output files
│ ├── sample_input.mp4 # Recorded dashcam footage
│ └── detection_log.csv # Detection results log
│
└── demo/ # Demo video
└── demo_video.mp4


---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt

python src/main.py



