import os

# Disable FFmpeg frame buffering before OpenCV initializes
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# Import HandTracker first so the Python 3.14 ctypes patch runs before MediaPipe loads
from hand_tracker import HandTracker
from neural_network import ASLNeuralNetwork
from letters import WordBuilder

import time
import threading
from pathlib import Path
import cv2
from dotenv import load_dotenv


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


def draw_hud(frame, mode, status_msg, display_sign, display_conf, word_builder):
    """Renders the top status/confidence panel and bottom word-builder banner."""
    frame_h, frame_w = frame.shape[:2]
    hud_color = (0, 255, 0) if mode == "PREDICT" else (0, 165, 255)

    # Top-Left Status & Prediction HUD
    cv2.rectangle(frame, (15, 15), (520, 130), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, 15), (520, 130), hud_color, 2)

    cv2.putText(frame, f"MODE: {mode}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, hud_color, 2)
    if mode == "PREDICT":
        sign_text = f"Sign: {display_sign} ({int(display_conf * 100)}%)" if display_sign != "--" else "Sign: --"
        cv2.putText(frame, sign_text, (260, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        bar_w = int(470 * word_builder.hold_progress)
        cv2.rectangle(frame, (30, 62), (500, 80), (60, 60, 60), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (30, 62), (30 + bar_w, 80), (0, 255, 0), -1)

    cv2.putText(frame, status_msg[:58], (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

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
        "[TAB] Switch Mode | [SPACE] Space | [BKSP] Delete | [C] Clear | [Q] Quit"
        if mode == "PREDICT"
        else "HOLD SIGN & PRESS [A-Z] TO RECORD 30 SAMPLES | [TAB] Train & Predict"
    )
    cv2.putText(frame, controls, (30, frame_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 220, 255), 1)


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

    mode = "PREDICT"
    status_msg = nn.status_message
    record_label = None
    samples_remaining = 0
    needs_retrain = False

    stream = LatestFrameStream(rtsp_url)
    last_frame_id = -1
    last_timestamp_ms = 0

    try:
        while True:
            ret, frame, frame_id = stream.read()
            if not ret or frame is None:
                print("Waiting for RTSP stream...")
                time.sleep(0.05)
                continue

            # Skip redundant inference if no new camera frame has arrived yet
            if frame_id == last_frame_id:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue
            last_frame_id = frame_id

            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (int(width * 0.5), int(height * 0.5)))

            # MediaPipe VIDEO mode requires strictly increasing timestamps
            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(last_timestamp_ms + 1, now_ms)
            last_timestamp_ms = timestamp_ms

            hand, features = tracker.process_frame(frame, timestamp_ms)

            display_sign = "--"
            display_conf = 0.0

            if hand is not None:
                tracker.draw_overlay(frame, hand, word_builder.trail_points)

                if mode == "RECORD" and samples_remaining > 0 and record_label:
                    nn.save_sample(record_label, features)
                    samples_remaining -= 1
                    needs_retrain = True
                    status_msg = f"Recording '{record_label}': {30 - samples_remaining}/30 samples"
                    if samples_remaining == 0:
                        status_msg = f"Saved 30 samples for '{record_label}'. Press another key or TAB."

                elif mode == "PREDICT":
                    pred_letter, display_conf = nn.predict(features)
                    display_sign = word_builder.update(hand, pred_letter, display_conf)
            else:
                word_builder.reset_timer()
                word_builder.clear_motion()

            draw_hud(frame, mode, status_msg, display_sign, display_conf, word_builder)
            cv2.imshow("ASL Neural Network & Word Builder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 9:  # TAB
                if mode == "PREDICT":
                    mode = "RECORD"
                    word_builder.reset_timer()
                    status_msg = "Hold a sign and press A-Z to capture samples."
                else:
                    mode = "PREDICT"
                    if needs_retrain:
                        nn.train()
                        status_msg = nn.status_message
                        needs_retrain = False
                    else:
                        status_msg = "Switched to Predict mode."
            elif mode == "RECORD":
                if (ord("a") <= key <= ord("z")) or (ord("A") <= key <= ord("Z")):
                    record_label = chr(key).upper()
                    samples_remaining = 30
            elif mode == "PREDICT":
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