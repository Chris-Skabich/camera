import sys
import ctypes

# Python 3.13+ Windows C-runtime patch for MediaPipe (must run before importing mediapipe)
if sys.platform == "win32":
    _original_getattr = ctypes.CDLL.__getattr__

    def _patched_getattr(self, name):
        if name == "free":
            ucrt = ctypes.CDLL("ucrtbase")
            return _original_getattr(ucrt, "free")
        return _original_getattr(self, name)

    ctypes.CDLL.__getattr__ = _patched_getattr

from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Key finger landmarks (MCP, PIP, TIP) used to measure thumb placement in closed-fist signs
THUMB_TARGET_LANDMARKS = [5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
FINGER_JOINTS = [(6, 8), (10, 12), (14, 16), (18, 20)]
FEATURE_COUNT = 79


class HandTracker:
    """Wraps MediaPipe HandLandmarker and converts hand landmarks into 79 3D + geometric features."""

    def __init__(self, model_filename="hand_landmarker.task"):
        model_path = Path(__file__).resolve().parent / model_filename
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr, timestamp_ms):
        """Detects the primary hand and returns (raw_landmarks, normalized_79_features)."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None, None

        hand = result.hand_landmarks[0]
        features = self._extract_features(hand)
        return hand, features

    @staticmethod
    def _extract_features(hand):
        """
        Builds a scale-invariant 79-float vector:
          - 63 normalized 3D coordinates (x, y, z relative to wrist)
          - 12 3D distances from thumb tip (4) to index/middle/ring/pinky MCP, PIP, and TIP
          -  4 finger curl ratios (tip-to-wrist vs PIP-to-wrist distance)
        """
        wx, wy, wz = hand[0].x, hand[0].y, hand[0].z

        rel_pts = np.array(
            [[lm.x - wx, lm.y - wy, lm.z - wz] for lm in hand],
            dtype=np.float32,
        )

        # Normalize using 2D hand span so Z depth preserves proportional scale
        max_xy = float(np.max(np.abs(rel_pts[:, :2])))
        scale = max_xy if max_xy > 1e-6 else 1.0
        norm_pts = rel_pts / scale

        # 1. Flattened 3D coordinates (63 features)
        coord_features = norm_pts.flatten()

        # 2. 3D Euclidean distances from thumb tip (landmark 4) to key finger joints (12 features)
        thumb_tip = norm_pts[4]
        thumb_dists = np.array(
            [np.linalg.norm(thumb_tip - norm_pts[idx]) for idx in THUMB_TARGET_LANDMARKS],
            dtype=np.float32,
        )

        # 3. Finger curl ratios to separate tight curls (E) from flat folds (A/S) (4 features)
        curl_ratios = np.array(
            [
                np.linalg.norm(norm_pts[tip]) / max(float(np.linalg.norm(norm_pts[pip])), 1e-6)
                for pip, tip in FINGER_JOINTS
            ],
            dtype=np.float32,
        )

        return np.concatenate([coord_features, thumb_dists, curl_ratios])

    @staticmethod
    def draw_overlay(frame, hand, trail_points=None):
        """Draws the 21 skeletal landmarks, bounding box, and optional J/Z motion trail."""
        frame_h, frame_w = frame.shape[:2]
        x_coords = [int(lm.x * frame_w) for lm in hand]
        y_coords = [int(lm.y * frame_h) for lm in hand]

        for x, y in zip(x_coords, y_coords):
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(
                frame,
                (x_coords[p1], y_coords[p1]),
                (x_coords[p2], y_coords[p2]),
                (0, 255, 0),
                2,
            )

        if trail_points and len(trail_points) > 1:
            pts = [(int(px * frame_w), int(py * frame_h)) for px, py in trail_points]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), 3)

        x_min, y_min = max(0, min(x_coords) - 20), max(0, min(y_coords) - 20)
        x_max, y_max = min(frame_w, max(x_coords) + 20), min(frame_h, max(y_coords) + 20)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)