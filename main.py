# Main orchestration loop
# This executes the unified prototype frame loop. It runs on your webcam, checks defined zones, estimates gaze orientation, manages employee persistence, and outputs live metrics.
# main.py
# main.py
import cv2
import time
from config.settings import VIDEO_SOURCE
from core.zones import ZoneChecker
from core.gaze import HybridGazeEstimator
from core.reid import EmployeeReIDDatabase

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Force high resolution matching drawer settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    zone_checker = ZoneChecker()
    gaze_engine = HybridGazeEstimator()
    reid_db = EmployeeReIDDatabase()

    # Read first clean frame to run perspective alignment check
    ret, initial_frame = cap.read()
    if ret:
        print("[INFO] Running homography alignment check against reference...")
        zone_checker.align_zones_to_current_view(initial_frame)
        print("[INFO] Alignment complete.")

    last_time = time.time()
    window_name = "Production Desk Analytics Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        dt = time.time() - last_time
        last_time = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Render zones as clean, translucent fills
        overlay = frame.copy()
        for idx, (zone_name, poly) in enumerate(zone_checker.aligned_zones.items()):
            color = (255, 100, 0) if idx % 2 == 0 else (0, 150, 255)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, color, 2)
            cv2.putText(frame, zone_name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        alpha = 0.15  # Subtle highlight transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Gaze Process
        gaze_results = gaze_engine.process_gaze(frame_rgb)

        for detection in gaze_results:
            is_looking = detection["is_looking"]
            yaw = detection["yaw"]
            nose_pos = detection["nose_coords"]

            active_zone = zone_checker.check_position(nose_pos)

            # Crop Face
            x, y = nose_pos
            x1, y1 = max(0, x - 80), max(0, y - 100)
            x2, y2 = min(w, x + 80), min(h, y + 100)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                profile_id, is_new = reid_db.identify_person_by_embedding(face_crop)

                if active_zone:
                    reid_db.log_dwell_time(profile_id, active_zone, dt)
                reid_db.log_gaze_event(profile_id, is_looking)

                # Draw bounding box and label metadata
                box_color = (0, 255, 0) if is_looking else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                gaze_txt = "LOOKING" if is_looking else "AWAY"
                accum_dwell, gaze_looks = reid_db.get_metrics(profile_id, active_zone)
                
                cv2.putText(frame, f"{profile_id} ({active_zone})", (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                cv2.putText(frame, f"Gaze: {gaze_txt} | Yaw: {yaw:.1f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Time: {accum_dwell:.1f}s | Looks: {gaze_looks}", (x1, y2 + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()