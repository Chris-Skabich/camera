# Real-Time ASL Neural Network & Word Builder

A real-time American Sign Language (ASL) fingerspelling recognizer and word builder powered by **MediaPipe**, a **scikit-learn Multi-Layer Perceptron (MLP) Neural Network**, and a **kinematic trajectory tracker**. Designed to run on a PC pulling a live RTSP stream from a Hanwha Wisenet QNV-7080R IP camera.

---

## How It Works (Under the Hood)

### 1. Zero-Latency RTSP Ingestion (`LatestFrameStream`)
Standard OpenCV RTSP captures queue incoming frames in an internal buffer. Whenever neural network inference takes slightly longer than the camera's frame interval, the display falls seconds behind live movement. To prevent this:
- FFmpeg is configured via `OPENCV_FFMPEG_CAPTURE_OPTIONS` with `nobuffer` and `low_delay` flags.
- A background daemon thread (`LatestFrameStream`) continuously drains the RTSP socket so the main loop always processes the single most recent frame.

### 2. 79-Feature 3D & Geometric Hand Representation (`HandTracker`)
Raw pixel coordinates fail whenever you move closer to or farther from the camera, and 2D `(x, y)` coordinates alone easily confuse closed-fist signs (**A, E, M, N, S, T**). Each frame, MediaPipe extracts 21 3D hand landmarks, which `HandTracker` normalizes relative to the wrist (`landmark[0]`) and scales by 2D palm span into a **79-float feature vector**:
- **63 Coordinate Features:** Normalized 3D `(x, y, z)` positions for all 21 joints.
- **12 Thumb-to-Finger Distances:** 3D Euclidean distances from the thumb tip (`landmark[4]`) to the MCP, PIP, and TIP joints of the index, middle, ring, and pinky fingers—capturing exact thumb tuck placement to distinguish **A, E, M, N, S, and T**.
- **4 Finger Curl Ratios:** Tip-to-wrist distance divided by PIP-to-wrist distance for each finger, separating tight curls (**E**) from flat folds (**A, S**).

### 3. Feedforward Neural Network (`ASLNeuralNetwork`)
Rather than relying on rigid hardcoded angle rules, `neural_network.py` builds a `scikit-learn` pipeline (`StandardScaler` $\rightarrow$ `MLPClassifier`) with two hidden layers (`128` $\rightarrow$ `64` ReLU neurons), Adam optimization, and L2 regularization (`alpha=0.001`). Samples can be recorded live from the video stream and trained in under a second.

### 4. Hybrid Static + Dynamic Word Builder (`WordBuilder`)
ASL fingerspelling mixes **24 static poses** with **2 motion gestures (J and Z)**, which `letters.py` handles simultaneously:
- **Static Dwell Locking:** When a static letter is predicted at $\ge 75\%$ confidence, a green progress bar charges over **2.0 seconds** before locking the letter into the bottom word buffer, followed by a **1.5-second cooldown** to prevent accidental duplicates.
- **Kinematic Motion Tracking (`J` & `Z`):** A 2.5-second rolling buffer tracks smoothed, palm-normalized fingertip coordinates with a 5-frame grace tolerance for noisy frames:
  - **Letter `J`:** Triggered when an isolated extended pinky (`I` pose) sweeps downward and hooks sideways. Active movement automatically pauses the static `I` dwell timer.
  - **Letter `Z`:** Triggered when an isolated extended index finger (`1` pose) traces a 3-segment zigzag (horizontal $\rightarrow$ diagonal down $\rightarrow$ horizontal) in either left-to-right or right-to-left orientation.

### 5. Interactive Stream Controls & ASL Reference Chart
Two clickable HUD buttons in the top-right corner of the video window streamline live practice and retraining without leaving the stream:
- **ASL Reference Chart (`ASL CHART` / `HIDE CHART`):** Docks a full ASL alphabet visual guide side-by-side on the right edge of the window so it never covers your hand or HUD. It automatically loads any chart image (`.jpg`, `.png`, `.webp`, `.bmp`) saved in `asl/` (or auto-downloads an illustrated chart on first click, falling back to an offline A–Z text guide if disconnected).
- **One-Click Dataset Reset (`CLEAR CSV`):** Uses a two-step safety confirmation (`CLEAR CSV` $\rightarrow$ `CONFIRM?` within 3 seconds) to delete `asl_dataset.csv` and `asl_nn_model.pkl`, clear the in-memory model, and jump straight into **RECORD** mode so you can retrain from scratch.

### 6. Python 3.13 / 3.14 Windows C-Runtime Patch
On Windows, Python 3.13+ isolates the C runtime such that MediaPipe's internal `ctypes` loader fails to locate `free` (`AttributeError: function 'free' not found`). `hand_tracker.py` intercepts `ctypes.CDLL.__getattr__` before MediaPipe initializes and redirects `free` lookups to Windows' Universal C Runtime (`ucrtbase`).

---

## Project Structure

```text
asl/
├── sign_language.py      # Main RTSP loop, threaded frame reader, HUD/buttons, and key controls
├── hand_tracker.py       # Python 3.14 ctypes patch, MediaPipe wrapper, and 79-feature extraction
├── neural_network.py     # CSV dataset I/O, dataset reset, MLP neural network training, and inference
├── letters.py            # Dwell-time word builder and J/Z kinematic trajectory recognizer
├── hand_landmarker.task  # Google MediaPipe HandLandmarker model binary
├── asl_alphabet.png      # Local or auto-downloaded ASL alphabet reference chart image
├── asl_dataset.csv       # Auto-generated training samples (created when you record signs)
└── asl_nn_model.pkl      # Auto-generated serialized neural network weights
```

---

## Setup & Running

### 1. Install Dependencies
From your repository root or `asl/` directory:
```powershell
pip install opencv-python mediapipe scikit-learn joblib numpy python-dotenv
```

### 2. Download the MediaPipe Hand Landmarker Model
Inside the `asl/` folder, download Google's pre-trained `.task` model:
```powershell
Invoke-WebRequest -Uri "[https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)" -OutFile "hand_landmarker.task"
```

### 3. Configure Environment Variables
Ensure `.env` exists in the parent directory (`camera/.env`) with your camera's RTSP stream URL:
```text
RTSP_URL=rtsp://admin:YOUR_PASSWORD@169.254.166.100:554/profile2/media.smp
```

### 4. Launch the Application
```powershell
cd asl
python .\sign_language.py
```

---

## Controls & Workflow

### Keyboard Shortcuts

| Mode | Key | Action |
| :--- | :--- | :--- |
| **Global** | `TAB` | Toggle between **PREDICT** mode and **RECORD** mode (automatically retrains the NN when returning to Predict) |
| **Global** | `ESC` | Exit the application |
| **RECORD** | `A` – `Z` | Capture a burst of **30 normalized 79-feature samples** for that letter *(tilt your hand slightly during capture for better generalization)* |
| **PREDICT** | `SPACE` | Append a space to the word buffer |
| **PREDICT** | `BACKSPACE` | Delete the last character in the word buffer |
| **PREDICT** | `C` | Clear the entire word buffer |
| **PREDICT** | `Q` | Exit the application |

### On-Screen Mouse Buttons

| Button | Action |
| :--- | :--- |
| **`ASL CHART` / `HIDE CHART`** | Dock or hide the side-by-side ASL alphabet reference chart next to the live stream |
| **`CLEAR CSV` $\rightarrow$ `CONFIRM?`** | Click once to arm, then click **`CONFIRM?`** within 3 seconds to delete `asl_dataset.csv` and `asl_nn_model.pkl` and switch into **RECORD** mode for fresh retraining |

> **Note:** You do not need to record **`J`** or **`Z`** in Record mode—only record static letters (including **`I`**), and the motion tracker will automatically recognize **`J`** and **`Z`** gestures on the fly.