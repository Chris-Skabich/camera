from collections import deque
import math
import time


class WordBuilder:
    """Stabilizes static signs via dwell time and detects dynamic motion signs (J and Z)."""

    def __init__(self, hold_seconds=2.0, cooldown_seconds=1.5, conf_threshold=0.75):
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.conf_threshold = conf_threshold

        self.word = ""
        self.candidate_letter = None
        self.candidate_start = 0.0
        self.last_locked_time = 0.0
        self.hold_progress = 0.0

        # Rolling history of recent fingertip positions for J and Z motion detection
        self.motion_window = 2.5
        self.motion_history = deque(maxlen=90)
        self.trail_points = []
        self.missed_pose_frames = 0

    def update(self, hand, predicted_letter, confidence):
        """
        Evaluates dynamic motion signs (J, Z) first, then static dwell signs.
        Returns the active display sign ('A'-'Z' or '--').
        """
        now = time.time()

        # 1. Check dynamic motion signs (J and Z) when outside cooldown
        if hand is not None and (now - self.last_locked_time >= self.cooldown_seconds):
            motion_letter, is_moving = self._update_motion(hand, now)
            if motion_letter is not None:
                self.word += motion_letter
                self.last_locked_time = now
                self.reset_timer()
                self.clear_motion()
                return motion_letter

            # Prevent static 'I' from locking while actively sweeping a 'J'
            if is_moving:
                self.reset_timer()
                return predicted_letter if (predicted_letter and confidence >= self.conf_threshold) else "--"

        # 2. Standard static sign dwell logic
        if predicted_letter is None or confidence < self.conf_threshold:
            self.reset_timer()
            return "--"

        if now - self.last_locked_time >= self.cooldown_seconds:
            if predicted_letter == self.candidate_letter:
                self.hold_progress = min(1.0, (now - self.candidate_start) / self.hold_seconds)
                if self.hold_progress >= 1.0:
                    self.word += predicted_letter
                    self.last_locked_time = now
                    self.reset_timer()
                    self.clear_motion()
            else:
                self.candidate_letter = predicted_letter
                self.candidate_start = now
                self.hold_progress = 0.0

        return predicted_letter

    def _update_motion(self, hand, now):
        """Tracks smoothed pinky/index paths normalized by palm size to detect J and Z."""
        while self.motion_history and (now - self.motion_history[0]["t"] > self.motion_window):
            self.motion_history.popleft()

        wrist = hand[0]
        palm_size = math.hypot(hand[9].x - wrist.x, hand[9].y - wrist.y)
        if palm_size < 1e-4:
            return None, False

        def is_extended(tip_idx, pip_idx):
            tip_d = math.hypot(hand[tip_idx].x - wrist.x, hand[tip_idx].y - wrist.y)
            pip_d = math.hypot(hand[pip_idx].x - wrist.x, hand[pip_idx].y - wrist.y)
            return tip_d > pip_d * 1.06

        def is_folded(tip_idx, pip_idx):
            tip_d = math.hypot(hand[tip_idx].x - wrist.x, hand[tip_idx].y - wrist.y)
            pip_d = math.hypot(hand[pip_idx].x - wrist.x, hand[pip_idx].y - wrist.y)
            return tip_d < pip_d * 1.22

        pinky_only = (
            is_extended(20, 18)
            and is_folded(8, 6)
            and is_folded(12, 10)
            and is_folded(16, 14)
        )
        index_only = (
            is_extended(8, 6)
            and is_folded(12, 10)
            and is_folded(16, 14)
            and is_folded(20, 18)
        )

        # Grace buffer: tolerate up to 5 noisy frames mid-stroke before resetting
        if not (pinky_only or index_only):
            self.missed_pose_frames += 1
            if self.missed_pose_frames > 5:
                self.clear_motion()
            return None, False

        self.missed_pose_frames = 0
        self.motion_history.append({
            "t": now,
            "pinky_only": pinky_only,
            "index_only": index_only,
            "px_norm": hand[20].x / palm_size,
            "py_norm": hand[20].y / palm_size,
            "ix_norm": hand[8].x / palm_size,
            "iy_norm": hand[8].y / palm_size,
            "px_screen": hand[20].x,
            "py_screen": hand[20].y,
            "ix_screen": hand[8].x,
            "iy_screen": hand[8].y,
        })

        if pinky_only:
            seq = [p for p in self.motion_history if p["pinky_only"]]
            self.trail_points = [(p["px_screen"], p["py_screen"]) for p in seq]
            pts = self._smooth_points([(p["px_norm"], p["py_norm"]) for p in seq])
            is_moving = self._recent_displacement(pts) > 0.12
            if self._match_j(pts):
                return "J", is_moving
            return None, is_moving

        if index_only:
            seq = [p for p in self.motion_history if p["index_only"]]
            self.trail_points = [(p["ix_screen"], p["iy_screen"]) for p in seq]
            pts = self._smooth_points([(p["ix_norm"], p["iy_norm"]) for p in seq])
            is_moving = self._recent_displacement(pts) > 0.12
            if self._match_z(pts):
                return "Z", is_moving
            return None, is_moving

        return None, False

    @staticmethod
    def _smooth_points(pts):
        """Applies a 3-point moving average to eliminate MediaPipe landmark jitter."""
        if len(pts) < 3:
            return pts
        smoothed = [pts[0]]
        for i in range(1, len(pts) - 1):
            sx = (pts[i - 1][0] + pts[i][0] + pts[i + 1][0]) / 3.0
            sy = (pts[i - 1][1] + pts[i][1] + pts[i + 1][1]) / 3.0
            smoothed.append((sx, sy))
        smoothed.append(pts[-1])
        return smoothed

    @staticmethod
    def _recent_displacement(pts):
        """Returns normalized fingertip travel over the last 4 tracked frames."""
        if len(pts) < 4:
            return 0.0
        return math.hypot(pts[-1][0] - pts[-4][0], pts[-1][1] - pts[-4][1])

    @staticmethod
    def _match_j(pts):
        """Detects a downward pinky drop followed by a sideways hook at the lowest point."""
        n = len(pts)
        if n < 8:
            return False

        # Locate the actual lowest point (maximum Y) rather than assuming the halfway frame
        bottom_idx = max(range(2, n - 2), key=lambda i: pts[i][1])
        x0, y0 = pts[0]
        xb, yb = pts[bottom_idx]
        x1, y1 = pts[-1]

        dy_down = yb - y0
        dx_down = abs(xb - x0)
        dx_hook = abs(x1 - xb)
        dy_hook = yb - y1  # Positive when the hook curves back upward

        downward_ok = dy_down > 0.28 and dy_down > dx_down * 0.75
        hook_ok = dx_hook > 0.20 and -0.10 <= dy_hook <= 0.35
        return downward_ok and hook_ok

    @staticmethod
    def _match_z(pts):
        """Searches for a 3-segment Z zigzag (horizontal -> diagonal down -> horizontal)."""
        n = len(pts)
        if n < 10:
            return False

        x0, y0 = pts[0]
        x3, y3 = pts[-1]

        # Test both left-to-right and right-to-left orientations (mirrored vs unmirrored view)
        for direction in (1, -1):
            for c1 in range(2, n - 5):
                x1, y1 = pts[c1]
                dx1, dy1 = x1 - x0, y1 - y0
                if not ((direction * dx1) > 0.26 and abs(dy1) < abs(dx1) * 0.75):
                    continue

                for c2 in range(c1 + 2, n - 2):
                    x2, y2 = pts[c2]
                    dx2, dy2 = x2 - x1, y2 - y1
                    if not ((direction * dx2) < -0.24 and dy2 > 0.22):
                        continue

                    dx3, dy3 = x3 - x2, y3 - y2
                    if (direction * dx3) > 0.26 and abs(dy3) < abs(dx3) * 0.75:
                        return True

        return False

    def reset_timer(self):
        self.candidate_letter = None
        self.hold_progress = 0.0

    def clear_motion(self):
        self.motion_history.clear()
        self.trail_points = []
        self.missed_pose_frames = 0

    def add_space(self):
        self.word += " "

    def backspace(self):
        self.word = self.word[:-1]

    def clear(self):
        self.word = ""