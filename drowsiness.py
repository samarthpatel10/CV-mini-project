import cv2
import csv
import time
import threading
import platform
import os
import urllib.request
from datetime import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- Settings (tune these) ----------------
EAR_THRESHOLD = 0.21         # below this = eye closed
EYES_CLOSED_SECONDS = 1.5
MAR_THRESHOLD = 0.6          # above this = yawn
YAWN_SECONDS = 1.5
HEAD_ANGLE_THRESHOLD = 15    # degrees away from your normal head position
HEAD_DOWN_SECONDS = 1.0
CALIBRATION_FRAMES = 30      # frames used to learn your normal head pose
LOG_FILE = "drowsiness_log.csv"
CAMERA_INDEX = 0
WINDOW_NAME = "Driver Drowsiness Alert"
# --------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 13, 308, 14]

# Head pose: nose tip, chin, eye corner (image-left), eye corner (image-right),
# mouth corner (image-left), mouth corner (image-right)
POSE_IDX = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

alarm_active = False


# ---------------- Model download ----------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return True
    print("Downloading face_landmarker.task (one time, ~4 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Could not download the model: {e}")
        print(f"Download it manually from:\n{MODEL_URL}\n"
              f"and save it as:\n{MODEL_PATH}")
        return False


# ---------------- CSV logger ----------------
class EventLogger:
    def __init__(self, path):
        path = os.path.join(BASE_DIR, path)
        new_file = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new_file:
            self.w.writerow(["timestamp", "event", "ear", "mar", "head_pitch"])
            self.f.flush()

    def log(self, event, ear=0.0, mar=0.0, pitch=0.0):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.w.writerow([ts, event, f"{ear:.3f}", f"{mar:.3f}", f"{pitch:.1f}"])
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------- Alarm ----------------
def play_alarm():
    global alarm_active
    if alarm_active:
        return
    alarm_active = True

    def _beep():
        global alarm_active
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                winsound.Beep(2500, 600)
            elif system == "Darwin":
                os.system("afplay /System/Library/Sounds/Funk.aiff")
            else:
                print("\a", end="", flush=True)
                time.sleep(0.6)
        finally:
            alarm_active = False

    threading.Thread(target=_beep, daemon=True).start()


# ---------------- Geometry helpers ----------------
def get_points(landmarks, indices, w, h):
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])


def eye_aspect_ratio(p):
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    hd = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * hd)


def mouth_aspect_ratio(p):
    return np.linalg.norm(p[1] - p[3]) / np.linalg.norm(p[0] - p[2])


def head_pitch(landmarks, w, h):
    """Return head pitch (up/down tilt) in degrees, or None on failure."""
    img_pts = get_points(landmarks, POSE_IDX, w, h).astype(np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, img_pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    pitch = angles[0]
    # Fix the +/-180 flip that solvePnP sometimes produces
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch


# ---------------- Main ----------------
def main():
    if not ensure_model():
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    logger = EventLogger(LOG_FILE)
    logger.log("SESSION_START")

    eyes_closed_start = yawn_start = head_start = None
    drowsy_logged = yawn_logged = head_logged = False
    drowsy_count = yawn_count = head_count = 0

    calib_pitches = []
    baseline_pitch = None

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    start_time = time.time()
    last_ts = -1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("Running. Click the video window and press q or Esc to quit.")

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Timestamps must strictly increase in VIDEO mode
                ts_ms = int((time.time() - start_time) * 1000)
                if ts_ms <= last_ts:
                    ts_ms = last_ts + 1
                last_ts = ts_ms
                results = landmarker.detect_for_video(mp_image, ts_ms)

                status, color = "AWAKE", (0, 200, 0)
                ear = mar = 0.0
                pitch = 0.0
                alert = False

                if results.face_landmarks:
                    lm = results.face_landmarks[0]
                    left = get_points(lm, LEFT_EYE, w, h)
                    right = get_points(lm, RIGHT_EYE, w, h)
                    mouth = get_points(lm, MOUTH, w, h)

                    ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
                    mar = mouth_aspect_ratio(mouth)
                    p = head_pitch(lm, w, h)
                    pitch = p if p is not None else 0.0

                    for pts in (left, right, mouth):
                        for (x, y) in pts:
                            cv2.circle(frame, (int(x), int(y)), 2,
                                       (0, 255, 255), -1)

                    now = time.time()

                    # --- Head pose calibration ---
                    if baseline_pitch is None:
                        if p is not None:
                            calib_pitches.append(p)
                        status, color = "CALIBRATING - LOOK STRAIGHT", (255, 255, 0)
                        if len(calib_pitches) >= CALIBRATION_FRAMES:
                            baseline_pitch = float(np.median(calib_pitches))
                    else:
                        deviation = abs(pitch - baseline_pitch)

                        # Slowly adapt baseline only while head is near normal
                        if deviation < HEAD_ANGLE_THRESHOLD / 2:
                            baseline_pitch = 0.99 * baseline_pitch + 0.01 * pitch

                        # --- Head nodding / drooping ---
                        if deviation > HEAD_ANGLE_THRESHOLD:
                            if head_start is None:
                                head_start = now
                            if now - head_start >= HEAD_DOWN_SECONDS:
                                status, color = "HEAD DOWN! WAKE UP!", (0, 0, 255)
                                alert = True
                                if not head_logged:
                                    head_count += 1
                                    head_logged = True
                                    logger.log("HEAD_DOWN", ear, mar, pitch)
                        else:
                            head_start = None
                            head_logged = False

                        # --- Eye closure ---
                        if ear < EAR_THRESHOLD:
                            if eyes_closed_start is None:
                                eyes_closed_start = now
                            if now - eyes_closed_start >= EYES_CLOSED_SECONDS:
                                status, color = "DROWSY! WAKE UP!", (0, 0, 255)
                                alert = True
                                if not drowsy_logged:
                                    drowsy_count += 1
                                    drowsy_logged = True
                                    logger.log("DROWSY", ear, mar, pitch)
                        else:
                            eyes_closed_start = None
                            drowsy_logged = False

                        # --- Yawn ---
                        if mar > MAR_THRESHOLD:
                            if yawn_start is None:
                                yawn_start = now
                            if now - yawn_start >= YAWN_SECONDS:
                                if not alert:
                                    status, color = ("YAWNING - TAKE A BREAK",
                                                     (0, 165, 255))
                                if not yawn_logged:
                                    yawn_count += 1
                                    yawn_logged = True
                                    logger.log("YAWN", ear, mar, pitch)
                        else:
                            yawn_start = None
                            yawn_logged = False
                else:
                    status, color = "NO FACE DETECTED", (128, 128, 128)
                    eyes_closed_start = yawn_start = head_start = None

                if alert:
                    play_alarm()
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

                # --- HUD ---
                cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
                cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)
                cv2.putText(frame,
                            f"EAR: {ear:.2f}  MAR: {mar:.2f}  Pitch: {pitch:.0f}",
                            (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.putText(frame,
                            f"Drowsy: {drowsy_count}  Yawns: {yawn_count}  "
                            f"Head-down: {head_count}",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.putText(frame, "Press Q or Esc to quit", (w - 220, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow(WINDOW_NAME, frame)

                # --- Quit handling ---
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):   # q, Q or Esc
                    break
                # Also quit when the window's X button is clicked
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        logger.log("SESSION_END")
        logger.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Closed.")


if __name__ == "__main__":
    main()