# WISENET CAMERA PROJECTS WITH OPENCV

A collection of computer vision and edge AI projects built in Python, using a Hanwha Wisenet QNV-7080R IP camera as a raw edge sensor.

## Project Overview
This repository serves as a learning sandbox for OpenCV, deep learning, and camera API integration. 

Hanwha's Q-Series cameras do not support onboard Open Platform applications (`.cap` files). To bypass this vendor limitation, these programs run externally on a connected PC. The software pulls high-resolution video via **RTSP** and controls the camera's internal hardware (like relays and day/night modes) via Hanwha's **SUNAPI** (HTTP API).

## Hardware & Stack
* **Camera:** Hanwha Wisenet QNV-7080R (4MP, IR, PoE)
* **Network:** Direct PC connection via PoE injector (APIPA `169.254.x.x` subnet)
* **Core Libraries:** `opencv-python`, `requests`, `python-dotenv`

## Project Roadmap
This repository will house multiple standalone programs, each focusing on a different computer vision concept:

* **[01] Stream Basics:** Connecting to the RTSP feed, handling credentials securely, and digitally zooming/resizing frames. *(Completed)*
* **[02] Object Detection:** Integrating YOLOv8 to identify and crop nocturnal wildlife using the camera's active IR illumination. *(Planned)*
* **[03] ALPR (Automated License Plate Recognition):** Utilizing optical character recognition to log vehicles entering the frame. *(Planned)*
* **[04] Posture Monitor:** Real-time skeletal tracking using MediaPipe to calculate and correct desk posture. *(Planned)*

## Getting Started

### 1. Prerequisites
**Hardware:**
* **Camera:** Hanwha Wisenet QNV-7080R (or similar Hanwha IP camera)
* **Power:** PoE (Power over Ethernet) Injector (e.g., Procet) 
* **Cables:** 2x standard Ethernet cables (one connecting your PC/router to the PoE injector's "Data In" port, and one connecting the injector's "PoE Out" port to the camera)

**Software:**
Ensure you have Python 3.8+ installed, then install the required core libraries:
```bash
pip install opencv-python requests python-dotenv