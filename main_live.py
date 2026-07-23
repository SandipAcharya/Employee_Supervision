# main_live.py
import cv2
import time
import os
import queue
import threading
import numpy as np
from ultralytics import YOLO
from core.zones import ZoneChecker
from core.gaze import HybridGazeEstimator
from core.reid import EmployeeReIDDatabase

# ==========================================
# Thread A: Non-Blocking Video Reader
# ==========================================
class VideoCaptureThreading:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame

    def stop(self):
        self.started = False
        self.cap.release()


# ==========================================
# Thread C: Asynchronous Biometric Worker
# ==========================================
class AsyncFaceProcessor:
    def __init__(self, reid_db):
        self.reid_db = reid_db
        self.input_queue = queue.Queue(maxsize=5)
        self.registry = {} # Thread-safe mapping: { yolo_track_id: str(profile_id) }
        self.registry_lock = threading.Lock()
        self.started = False

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.process_queue, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def submit_task(self, track_id, face_crop):
        try:
            self.input_queue.put_nowait((track_id, face_crop))
        except queue.Full:
            pass

    def process_queue(self):
        while self.started:
            try:
                track_id, face_crop = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            profile_id = f"Tracking_ID_{track_id}"
            if face_crop.size > 0:
                matched_id, _ = self.reid_db.identify_person_by_embedding(face_crop)
                if matched_id != "Unknown":
                    profile_id = matched_id

            with self.registry_lock:
                self.registry[track_id] = profile_id

            self.input_queue.task_done()

    def get_profile_id(self, track_id):
        with self.registry_lock:
            return self.registry.get(track_id, None)

    def stop(self):
        self.started = False


# ==========================================
# Thread B: Core Processing & UI Loop
# ==========================================
def main():
    video_stream = VideoCaptureThreading(src=0).start()
    
    print("[INFO] Loading YOLO model on CPU...")
    model = YOLO("yolov8n.pt").to("cpu")
    
    zone_checker = ZoneChecker()
    gaze_engine = HybridGazeEstimator()
    reid_db = EmployeeReIDDatabase()
    
    # Pre-load known employee photos
    reid_db.enroll_employees()
    
    # Background worker strictly processes heavy ArcFace ONNX identification
    face_worker = AsyncFaceProcessor(reid_db).start()

    # Track Re-ID request cool-downs to prevent queue congestion
    # Format: { track_id: float (timestamp) }
    reid_requests = {}

    last_time = time.time()
    window_name = "Enterprise Desk Analytics Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Perspective alignment check
    time.sleep(1.0)
    ret, initial_frame = video_stream.read()
    if ret:
        print("[INFO] Running homography alignment check against reference...")
        zone_checker.align_zones_to_current_view(initial_frame)
        print("[INFO] Alignment complete.")

    while True:
        ret, frame = video_stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        dt = time.time() - last_time
        last_time = time.time()

        h, w, _ = frame.shape

        # Distinct, vibrant colors for zones (BGR format)
        ZONE_COLORS = [(255, 100, 0), (0, 150, 255), (0, 200, 0), (200, 0, 255), (0, 255, 255)]

        # Render zones as clean, translucent fills
        overlay = frame.copy()
        for idx, (zone_name, poly) in enumerate(zone_checker.aligned_zones.items()):
            color = ZONE_COLORS[idx % len(ZONE_COLORS)]
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, color, 2)
            cv2.putText(frame, f"Zone: {zone_name}", tuple(poly[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # 1. Real-Time Gaze Estimation on FULL frame (solves focal length distortion)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_gaze_results = gaze_engine.process_gaze(frame_rgb)

        # Run YOLO CPU Tracking with high confidence to prevent hallucinated background artifacts
        results = model.track(frame, persist=True, classes=0, conf=0.65, verbose=False, device="cpu")
        
        active_track_ids = []
        zone_occupancy = {zone: 0 for zone in zone_checker.aligned_zones.keys()}

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                active_track_ids.append(track_id)
                x1, y1, x2, y2 = box
                
                # For CCTV/Edge deployments, checking the feet and chest provides the most accurate spatial tracking without false edge triggers
                points_to_check = [
                    (int((x1 + x2) / 2), y2),                               # Feet (Bottom Center - Best for floor plan mapping)
                    (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.3))         # Chest point (Bare minimum center mass)
                ]

                active_zone = None
                for pt in points_to_check:
                    active_zone = zone_checker.check_position(pt)
                    if active_zone:
                        break
                        
                if active_zone:
                    zone_occupancy[active_zone] += 1

                person_crop = frame[y1:y2, x1:x2]

                # Match Gaze Result to this bounding box
                is_looking = False
                curr_yaw, curr_pitch = 0.0, 0.0
                for gr in all_gaze_results:
                    nx, ny = gr["nose_coords"]
                    if x1 <= nx <= x2 and y1 <= ny <= y2:
                        is_looking = gr["is_looking"]
                        curr_yaw = gr["yaw"]
                        curr_pitch = gr["pitch"]
                        break

                # 2. Trigger Biometric Re-ID (Throttled Background task)
                now = time.time()
                profile_id = face_worker.get_profile_id(track_id)
                
                # Keep retrying if unknown or verifying!
                is_unknown = profile_id is None or "Verifying" in profile_id or "Tracking_ID" in profile_id
                
                if person_crop.size > 0 and is_unknown:
                    last_req = reid_requests.get(track_id, 0)
                    if (now - last_req) > 3.0:
                        face_worker.submit_task(track_id, person_crop)
                        reid_requests[track_id] = now
                        
                # 3. Log spatial dwell time (Independent of Gaze)
                if active_zone and profile_id and "Verifying" not in profile_id and profile_id != "Unknown":
                    face_worker.reid_db.log_dwell_time(profile_id, active_zone, dt)

                # 4. Log gaze look events and duration
                if profile_id and "Verifying" not in profile_id and profile_id != "Unknown":
                    face_worker.reid_db.log_gaze_event(profile_id, is_looking, dt)

                # 5. Safely retrieve metrics via our thread-locked database method
                accum_dwell, gaze_looks = face_worker.reid_db.get_metrics(profile_id, active_zone) if profile_id else (0.0, 0)

                # Fallback label during background computation
                if profile_id is None:
                    profile_id = f"Verifying_{track_id}..."

                # Draw bounding box and detail overlays
                box_color = (0, 255, 0) if is_looking else (0, 165, 255)
                
                # Draw sleek corner brackets for bounding box instead of solid lines
                l = 25; t = 2
                cv2.line(frame, (x1, y1), (x1 + l, y1), box_color, t); cv2.line(frame, (x1, y1), (x1, y1 + l), box_color, t)
                cv2.line(frame, (x2, y1), (x2 - l, y1), box_color, t); cv2.line(frame, (x2, y1), (x2, y1 + l), box_color, t)
                cv2.line(frame, (x1, y2), (x1 + l, y2), box_color, t); cv2.line(frame, (x1, y2), (x1, y2 - l), box_color, t)
                cv2.line(frame, (x2, y2), (x2 - l, y2), box_color, t); cv2.line(frame, (x2, y2), (x2, y2 - l), box_color, t)
                
                gaze_txt = "LOOKING" if is_looking else "AWAY"
                zone_status = active_zone if active_zone else "None"

                # Double-stroke text for perfect readability without blocky backgrounds
                # Black outline
                cv2.putText(frame, f"{profile_id} ({zone_status})", (x1 + 5, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Gaze: {gaze_txt} | Dwell: {accum_dwell:.1f}s | Yaw:{curr_yaw:.0f}", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                # Colored fill
                cv2.putText(frame, f"{profile_id} ({zone_status})", (x1 + 5, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Gaze: {gaze_txt} | Dwell: {accum_dwell:.1f}s | Yaw:{curr_yaw:.0f}", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Professional Top Bar HUD (Does not block camera view)
        overlay_panel = frame.copy()
        bar_h = 45
        cv2.rectangle(overlay_panel, (0, 0), (w, bar_h), (15, 15, 20), -1)
        cv2.line(overlay_panel, (0, bar_h), (w, bar_h), (100, 100, 100), 1)
        cv2.addWeighted(overlay_panel, 0.85, frame, 0.15, 0, frame)
        
        cv2.putText(frame, "EDGE ANALYTICS HUD", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"CROWD: {len(active_track_ids)}", (250, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        x_offset = 380
        for z_name, count in zone_occupancy.items():
            color = (0, 255, 0) if count > 0 else (100, 100, 100)
            cv2.putText(frame, f"{z_name.upper()}: {count}", (x_offset, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            x_offset += 150

        # Clean old keys from local cooldown arrays
        for tid in list(reid_requests.keys()):
            if tid not in active_track_ids:
                del reid_requests[tid]

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("[INFO] ESC pressed. Exiting...")
            reid_db.export_logs()
            break

    video_stream.stop()
    face_worker.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()