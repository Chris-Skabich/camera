import os

# Disable FFmpeg frame buffering before OpenCV initializes
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# Import HandTracker first so the Python 3.14 ctypes patch runs before MediaPipe loads
from hand_tracker import HandTracker
from neural_network import ASLNeuralNetwork
from letters import WordBuilder

import time
import threading
import urllib.request
from pathlib import Path
import numpy as np
import cv2
from dotenv import load_dotenv

WINDOW_NAME = "ASL Neural Network & Word Builder"

FALLBACK_CHEATSHEET = [
    "A: Fist, thumb on side",
    "B: 4 fingers up, thumb in",
    "C: Curved C shape",
    "D: Index up, others touch thumb",
    "E: Tight claw, thumb tucked",
    "F: Index+thumb OK, 3 up",
    "G: Index+thumb point side",
    "H: Index+middle point side",
    "I: Pinky up | J: Pinky J-hook",
    "K: V + thumb on middle",
    "L: Index up + thumb out",
    "M: Thumb under 3 fingers",
    "N: Thumb under 2 fingers",
    "O: All tips touch thumb",
    "P: K pointing downward",
    "Q: G pointing downward",
    "R: Index+middle crossed",
    "S: Fist, thumb across front",
    "T: Thumb under index finger",
    "U: Index+middle together",
    "V: Index+middle spread",
    "W: Index+middle+ring up",
    "X: Index bent like a hook",
    "Y: Thumb+pinky out | Z: Draw Z",
]


class LatestFrameStream:
    """Continuously drains the RTSP buffer in a background thread to eliminate lag."""

    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.ret, self.frame = self.cap.read()
        self.frame_id = 0
        self.running = True

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.frame_id += 1

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None, self.frame_id
            return True, self.frame.copy(), self.frame_id

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def load_reference_panel(target_height):
    """Finds any local ASL chart image or auto-downloads a visual ASL alphabet chart."""
    base_dir = Path(__file__).resolve().parent
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # 1. Search asl/ first, then parent camera/ folder for any saved image file
    candidates = []
    for folder in (base_dir, base_dir.parent):
        if folder.exists():
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in image_exts:
                    priority = 0 if any(k in path.stem.lower() for k in ("asl", "alphabet", "chart", "sign", "image")) else 1
                    candidates.append((priority, path))

    candidates.sort(key=lambda item: item[0])
    chart_path = candidates[0][1] if candidates else None

    # 2. If no image exists locally, auto-download a public-domain illustrated ASL chart
    if chart_path is None:
        download_target = base_dir / "asl_alphabet.png"
        chart_url = "https://superstarworksheets.com/wp-content/uploads/2024/03/ASLAlphabetChart-1.jpg"
        try:
            req = urllib.request.Request(chart_url, headers={"User-Agent": "ASLWordBuilder/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                download_target.write_bytes(resp.read())
            chart_path = download_target
        except Exception as exc:
            print(f"Could not download ASL chart image: {exc}")

    if chart_path is not None:
        img = cv2.imread(str(chart_path))
        if img is not None:
            h, w = img.shape[:2]
            new_w = max(220, int(w * (target_height / float(h))))
            return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

    # Offline fallback if no image is saved locally and download fails
    panel_w = 340
    panel = np.full((target_height, panel_w, 3), (24, 24, 24), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, target_height - 1), (90, 90, 90), 2)
    cv2.putText(panel, "ASL ALPHABET GUIDE", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    y_step = max(16, (target_height - 65) // len(FALLBACK_CHEATSHEET))
    for i, line in enumerate(FALLBACK_CHEATSHEET):
        y = 52 + i * y_step
        if y < target_height - 24:
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

    cv2.putText(
        panel,
        "Drop any .jpg/.png chart into asl/",
        (14, target_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (120, 200, 255),
        1,
    )
    return panel


def get_button_rects(frame_w):
    """Returns screen bounding boxes (x1, y1, x2, y2) for top-right clickable HUD buttons."""
    btn_w, btn_h = 155, 38
    x2 = frame_w - 15
    x1 = x2 - btn_w
    return {
        "chart": (x1, 15, x2, 15 + btn_h),
        "reset": (x1, 62, x2, 62 + btn_h),
    }


def draw_button(frame, rect, label, bg_color, border_color=(255, 255, 255), text_color=(255, 255, 255)):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2)


def draw_hud(frame, mode, status_msg, display_sign, display_conf, word_builder, show_chart, reset_armed):
    """Renders the status panel, clickable action buttons, and bottom word-builder banner."""
    frame_h, frame_w = frame.shape[:2]
    hud_color = (0, 255, 0) if mode == "PREDICT" else (0, 165, 255)
    buttons = get_button_rects(frame_w)

    # Top-Left Status & Prediction HUD
    hud_right = min(510, buttons["chart"][0] - 15)
    cv2.rectangle(frame, (15, 15), (hud_right, 130), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, 15), (hud_right, 130), hud_color, 2)

    cv2.putText(frame, f"MODE: {mode}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, hud_color, 2)
    if mode == "PREDICT":
        sign_text = f"Sign: {display_sign} ({int(display_conf * 100)}%)" if display_sign != "--" else "Sign: --"
        cv2.putText(frame, sign_text, (240, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

        bar_max = hud_right - 45
        bar_w = int(bar_max * word_builder.hold_progress)
        cv2.rectangle(frame, (30, 62), (30 + bar_max, 80), (60, 60, 60), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (30, 62), (30 + bar_w, 80), (0, 255, 0), -1)

    max_chars = max(25, (hud_right - 45) // 10)
    cv2.putText(frame, status_msg[:max_chars], (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # Top-Right Clickable Buttons
    chart_label = "HIDE CHART" if show_chart else "ASL CHART"
    chart_bg = (35, 110, 160) if show_chart else (35, 35, 35)
    draw_button(frame, buttons["chart"], chart_label, chart_bg, border_color=(100, 210, 255))

    if reset_armed:
        draw_button(frame, buttons["reset"], "CONFIRM?", (0, 0, 190), border_color=(0, 220, 255))
    else:
        draw_button(frame, buttons["reset"], "CLEAR CSV", (35, 35, 35), border_color=(90, 90, 255))

    # Bottom Word Builder Banner
    cv2.rectangle(frame, (15, frame_h - 95), (frame_w - 15, frame_h - 15), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, frame_h - 95), (frame_w - 15, frame_h - 15), (255, 255, 255), 2)

    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    cv2.putText(
        frame,
        f"Word: {word_builder.word}{cursor}",
        (30, frame_h - 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    controls = (
        "[TAB] Record Mode | [SPACE] Space | [BKSP] Delete | [C] Clear Word | [ESC] Quit"
        if mode == "PREDICT"
        else "HOLD SIGN & PRESS [A-Z] TO RECORD 30 SAMPLES | [TAB] Train & Predict"
    )
    cv2.putText(frame, controls, (30, frame_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 220, 255), 1)


def main():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        print("Error: RTSP_URL was not found in the .env file.")
        return

    tracker = HandTracker()
    nn = ASLNeuralNetwork()
    word_builder = WordBuilder(hold_seconds=2.0, cooldown_seconds=1.5)

    ui_state = {
        "mode": "PREDICT",
        "status_msg": nn.status_message,
        "record_label": None,
        "samples_remaining": 0,
        "needs_retrain": False,
        "show_chart": False,
        "reset_armed_until": 0.0,
        "frame_w": 960,
    }

    def on_mouse_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        buttons = get_button_rects(ui_state["frame_w"])

        # Toggle ASL Alphabet Reference Panel
        cx1, cy1, cx2, cy2 = buttons["chart"]
        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            ui_state["show_chart"] = not ui_state["show_chart"]
            return

        # Two-step Clear CSV Button
        rx1, ry1, rx2, ry2 = buttons["reset"]
        if rx1 <= x <= rx2 and ry1 <= y <= ry2:
            now = time.time()
            if now < ui_state["reset_armed_until"]:
                nn.reset_dataset()
                word_builder.reset_timer()
                word_builder.clear_motion()
                ui_state["mode"] = "RECORD"
                ui_state["record_label"] = None
                ui_state["samples_remaining"] = 0
                ui_state["needs_retrain"] = False
                ui_state["reset_armed_until"] = 0.0
                ui_state["status_msg"] = nn.status_message
            else:
                ui_state["reset_armed_until"] = now + 3.0
                ui_state["status_msg"] = "Click CONFIRM? within 3s to wipe CSV & retrain."

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

    stream = LatestFrameStream(rtsp_url)
    last_frame_id = -1
    last_timestamp_ms = 0
    cached_chart_panel = None
    cached_chart_height = -1

    try:
        while True:
            ret, frame, frame_id = stream.read()
            if not ret or frame is None:
                print("Waiting for RTSP stream...")
                time.sleep(0.05)
                continue

            if frame_id == last_frame_id:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            last_frame_id = frame_id

            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (int(width * 0.5), int(height * 0.5)))
            frame_h, frame_w = frame.shape[:2]
            ui_state["frame_w"] = frame_w

            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(last_timestamp_ms + 1, now_ms)
            last_timestamp_ms = timestamp_ms

            hand, features = tracker.process_frame(frame, timestamp_ms)

            display_sign = "--"
            display_conf = 0.0

            if hand is not None:
                tracker.draw_overlay(frame, hand, word_builder.trail_points)

                if ui_state["mode"] == "RECORD" and ui_state["samples_remaining"] > 0 and ui_state["record_label"]:
                    lbl = ui_state["record_label"]
                    nn.save_sample(lbl, features)
                    ui_state["samples_remaining"] -= 1
                    ui_state["needs_retrain"] = True
                    rem = ui_state["samples_remaining"]
                    ui_state["status_msg"] = f"Recording '{lbl}': {30 - rem}/30 samples"
                    if rem == 0:
                        ui_state["status_msg"] = f"Saved 30 samples for '{lbl}'. Press another key or TAB."

                elif ui_state["mode"] == "PREDICT":
                    pred_letter, display_conf = nn.predict(features)
                    display_sign = word_builder.update(hand, pred_letter, display_conf)
            else:
                word_builder.reset_timer()
                word_builder.clear_motion()

            reset_armed = time.time() < ui_state["reset_armed_until"]
            draw_hud(
                frame,
                ui_state["mode"],
                ui_state["status_msg"],
                display_sign,
                display_conf,
                word_builder,
                ui_state["show_chart"],
                reset_armed,
            )

            # Dock ASL alphabet chart image side-by-side when toggled ON
            if ui_state["show_chart"]:
                if cached_chart_panel is None or cached_chart_height != frame_h:
                    cached_chart_panel = load_reference_panel(frame_h)
                    cached_chart_height = frame_h
                display_frame = np.hstack((frame, cached_chart_panel))
            else:
                display_frame = frame

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 9:  # TAB
                if ui_state["mode"] == "PREDICT":
                    ui_state["mode"] = "RECORD"
                    word_builder.reset_timer()
                    ui_state["status_msg"] = "Hold a sign and press A-Z to capture samples."
                else:
                    ui_state["mode"] = "PREDICT"
                    if ui_state["needs_retrain"]:
                        nn.train()
                        ui_state["status_msg"] = nn.status_message
                        ui_state["needs_retrain"] = False
                    else:
                        ui_state["status_msg"] = "Switched to Predict mode."
            elif ui_state["mode"] == "RECORD":
                if (ord("a") <= key <= ord("z")) or (ord("A") <= key <= ord("Z")):
                    ui_state["record_label"] = chr(key).upper()
                    ui_state["samples_remaining"] = 30
            elif ui_state["mode"] == "PREDICT":
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    word_builder.add_space()
                elif key == 8:  # Backspace
                    word_builder.backspace()
                elif key == ord("c"):
                    word_builder.clear()
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()