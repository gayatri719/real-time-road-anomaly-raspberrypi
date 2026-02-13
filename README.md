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
- Unexpected obstacles

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
- Cooling setup (heat sink + fan)  
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

