"""
Attention Engine – Core computer vision logic using the new MediaPipe Tasks API
(mediapipe >= 0.10). Requires face_landmarker.task in the same directory.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import distance as dist
import time
import threading
from collections import deque

# ─── MediaPipe Tasks setup ────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

BaseOptions     = mp_python.BaseOptions
FaceLandmarker  = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

# ─── Landmark indices (MediaPipe 478-point mesh) ──────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# 3-D model points for head-pose (generic human face, in mm)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip       – lm 1
    (0.0, -330.0, -65.0),     # Chin           – lm 152
    (-225.0, 170.0, -135.0),  # Left eye right – lm 263
    (225.0, 170.0, -135.0),   # Right eye left – lm 33
    (-150.0, -150.0, -125.0), # Left mouth     – lm 287
    (150.0, -150.0, -125.0),  # Right mouth    – lm 57
], dtype="double")
MODEL_LM_IDS = [1, 152, 263, 33, 287, 57]

# ─── Thresholds ───────────────────────────────────────────────────────────────
EAR_THRESH     = 0.22
EAR_CONSEC     = 15
MAR_THRESH     = 0.65
YAW_THRESH     = 25
PITCH_THRESH   = 20
DISTRACTED_SEC = 3.0

ATTENTION_DECAY   = 0.985
ATTENTION_RECOVER = 0.992


def _eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)


def _mouth_aspect_ratio(landmarks, w, h):
    top    = np.array([landmarks[13].x * w, landmarks[13].y * h])
    bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])
    left   = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right  = np.array([landmarks[291].x * w, landmarks[291].y * h])
    return dist.euclidean(top, bottom) / (dist.euclidean(left, right) + 1e-6)


def _estimate_head_pose(landmarks, w, h):
    """Return (yaw, pitch, roll) in degrees."""
    image_pts = np.array(
        [(landmarks[m].x * w, landmarks[m].y * h) for m in MODEL_LM_IDS],
        dtype="double"
    )
    cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( R[2,1], R[2,2]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = 0.0
    return yaw, pitch, roll


class AttentionEngine:
    """Process each BGR video frame and emit a result dict."""

    def __init__(self, history_len=120):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = FaceLandmarker.create_from_options(options)

        self.attention_score  = 100.0
        self.ear_counter      = 0
        self.total_blinks     = 0
        self.distracted_since = None
        self.total_distracted = 0.0
        self.session_start    = time.time()
        self.last_frame_time  = self.session_start

        self.history           = deque(maxlen=history_len) # for real-time display (short window)
        self.session_history   = []                        # full history for report
        self.ear_history       = deque(maxlen=60)

        self.alert_drowsy       = False
        self.alert_looking_away = False
        self.alert_yawning      = False
        self.alert_no_face      = False

    # ── Public ───────────────────────────────────────────────────────────────

    def process(self, frame):
        """
        Process one BGR frame.
        Returns (annotated_frame, result_dict).
        """
        h, w = frame.shape[:2]
        now = time.time()
        session_sec = now - self.session_start
        dt = now - self.last_frame_time
        self.last_frame_time = now

        # Convert to MediaPipe Image
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection = self._detector.detect(mp_img)

        if not detection.face_landmarks:
            self._update_no_face(dt)
            result = self._build_result(0.0, 0.0, 0.0, 0.0, 0.0, False, session_sec)
            self._draw_no_face(frame, w, h)
            return frame, result

        self.alert_no_face = False
        lm = detection.face_landmarks[0]   # list of NormalizedLandmark

        ear_l = _eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        ear_r = _eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear   = (ear_l + ear_r) / 2.0

        mar   = _mouth_aspect_ratio(lm, w, h)
        yaw, pitch, roll = _estimate_head_pose(lm, w, h)

        self.ear_history.append(ear)

        self._update_blinks(ear)
        distracted = self._evaluate_attention(ear, yaw, pitch, mar, now, dt)

        self._draw_overlay(frame, lm, w, h, ear, distracted)

        result = self._build_result(ear, mar, yaw, pitch, roll, True, session_sec)
        return frame, result

    def reset_session(self):
        self.attention_score  = 100.0
        self.ear_counter      = 0
        self.total_blinks     = 0
        self.distracted_since = None
        self.total_distracted = 0.0
        self.session_start    = time.time()
        self.history.clear()
        self.session_history.clear()
        self.ear_history.clear()
        self.alert_drowsy       = False
        self.alert_looking_away = False
        self.alert_yawning      = False
        self.alert_no_face      = False

    # ── Private helpers ──────────────────────────────────────────────────────

    def _update_blinks(self, ear):
        if ear < EAR_THRESH:
            self.ear_counter += 1
        else:
            if self.ear_counter >= EAR_CONSEC:
                self.alert_drowsy = True
            elif 2 <= self.ear_counter < EAR_CONSEC:
                self.total_blinks += 1
                self.alert_drowsy  = False
            else:
                self.alert_drowsy  = False
            self.ear_counter = 0

    def _evaluate_attention(self, ear, yaw, pitch, mar, now, dt):
        distracted     = False
        looking_away   = abs(yaw) > YAW_THRESH or abs(pitch) > PITCH_THRESH
        eyes_closed    = ear < EAR_THRESH and self.ear_counter >= EAR_CONSEC
        yawning        = mar > MAR_THRESH

        self.alert_looking_away = looking_away
        self.alert_yawning      = yawning

        if looking_away or eyes_closed or yawning:
            if self.distracted_since is None:
                self.distracted_since = now
            if now - self.distracted_since >= DISTRACTED_SEC:
                distracted = True
                self.total_distracted += dt
                self.attention_score   = max(0.0, self.attention_score * ATTENTION_DECAY)
        else:
            self.distracted_since = None
            self.attention_score  = min(100.0, self.attention_score / ATTENTION_RECOVER)

        self.history.append(self.attention_score)
        self.session_history.append(self.attention_score)
        return distracted

    def _update_no_face(self, dt):
        self.alert_no_face  = True
        self.attention_score = max(0.0, self.attention_score * ATTENTION_DECAY)
        self.total_distracted += dt
        self.history.append(self.attention_score)
        self.session_history.append(self.attention_score)

    def _build_result(self, ear, mar, yaw, pitch, roll, face_detected, session_sec):
        s = self.attention_score
        avg_s = sum(self.session_history) / len(self.session_history) if self.session_history else s
        
        if s >= 80:   status = "Attentive"
        elif s >= 55: status = "Mildly Distracted"
        elif s >= 30: status = "Distracted"
        else:         status = "Very Distracted"

        return {
            "attention_score":    round(s, 1),
            "average_score":      round(avg_s, 1),
            "ear":                round(ear, 3),
            "mar":                round(mar, 3),
            "yaw":                round(yaw, 1),
            "pitch":              round(pitch, 1),
            "roll":               round(roll, 1),
            "blinks":             self.total_blinks,
            "distracted_sec":     round(self.total_distracted, 1),
            "session_sec":        round(session_sec, 1),
            "alert_drowsy":       self.alert_drowsy,
            "alert_looking_away": self.alert_looking_away,
            "alert_yawning":      self.alert_yawning,
            "alert_no_face":      self.alert_no_face,
            "status_label":       status,
            "face_detected":      face_detected,
            "history":            list(self.history),
            "session_full_history": list(self.session_history),
        }

    def _draw_overlay(self, frame, lm, w, h, ear, distracted):
        eye_color = (0, 200, 80) if ear >= EAR_THRESH else (0, 60, 255)
        for idx_list in [LEFT_EYE, RIGHT_EYE]:
            pts = np.array(
                [(int(lm[i].x * w), int(lm[i].y * h)) for i in idx_list],
                dtype=np.int32
            )
            cv2.polylines(frame, [pts], True, eye_color, 1)

        color = (0, 200, 80) if not distracted else (0, 60, 255)
        label = f"EAR:{ear:.2f}  Yaw:  Score:{self.attention_score:.0f}"
        cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.putText(frame, label, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def _draw_no_face(self, frame, w, h):
        cv2.rectangle(frame, (0, 0), (w, h), (30, 0, 0), 4)
        cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.putText(frame, "No face detected - Score falling",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 80, 255), 1, cv2.LINE_AA)
