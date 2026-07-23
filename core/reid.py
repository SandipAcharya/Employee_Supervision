# core/reid.py
import urllib.request
import os
import time
import numpy as np
import cv2
import onnxruntime as ort
import threading
import csv
import mediapipe as mp

class EmployeeReIDDatabase:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "w600k_mbf.onnx")
        
        # Download official lightweight 5.2MB ArcFace (MobileFaceNet) on first run
        self.download_url = "https://huggingface.co/WePrompt/buffalo_sc/resolve/main/w600k_mbf.onnx"
        self.ensure_model()

        print("[INFO] Initializing ArcFace Re-ID Engine on CPU...")
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Thread lock to prevent race conditions during concurrent accesses
        self.db_lock = threading.Lock()
        
        # Database Schema:
        # { profile_id: {
        #     "embedding": np.array,
        #     "gaze_look_count": int,
        #     "is_currently_looking": bool,  # State track for rising-edge trigger
        #     "dwell_times": { "zone_name": float },
        #     "last_seen": float
        # }}
        self.db = {}
        self.next_id_counter = 1
        self.similarity_threshold = 0.55
        
        # Initialize MediaPipe Face Detection to robustly crop faces before ArcFace
        # model_selection=0 is designed for faces within 2 meters (perfect for webcams/desks)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)

    def ensure_model(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Downloading pretrained ArcFace model from {self.download_url}...")
            try:
                req = urllib.request.Request(self.download_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(self.model_path, 'wb') as out_file:
                    out_file.write(response.read())
                print("[INFO] Model downloaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to download weights automatically: {e}")
                raise e

    def enroll_employees(self, enrollment_dir="enrollment"):
        """Loads reference images of employees to establish known identities using Multi-Shot averaging."""
        os.makedirs(enrollment_dir, exist_ok=True)
        print(f"[INFO] Looking for employee photos in '{enrollment_dir}' directory...")
        
        # Temporary dictionary to group embeddings by employee name
        person_embeddings = {}

        for item in os.listdir(enrollment_dir):
            item_path = os.path.join(enrollment_dir, item)
            
            # Case 1: Sub-directory for each employee (e.g., enrollment/Sandip/)
            if os.path.isdir(item_path):
                emp_name = item
                for filename in os.listdir(item_path):
                    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img = cv2.imread(os.path.join(item_path, filename))
                        if img is not None:
                            emb = self.extract_embedding(img)
                            if emb is not None:
                                person_embeddings.setdefault(emp_name, []).append(emb)
            
            # Case 2: Flat files with numbering (e.g., enrollment/Sandip_01.jpg)
            elif item.lower().endswith(('.jpg', '.png', '.jpeg')):
                basename = os.path.splitext(item)[0]
                # Strip numeric suffix if present (e.g., Sandip_01 -> Sandip)
                parts = basename.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    emp_name = '_'.join(parts[:-1])
                else:
                    emp_name = basename
                    
                img = cv2.imread(item_path)
                if img is not None:
                    emb = self.extract_embedding(img)
                    if emb is not None:
                        person_embeddings.setdefault(emp_name, []).append(emb)

        # Average embeddings for each person to create a robust Master Template
        count = 0
        for name, embs in person_embeddings.items():
            if not embs:
                continue
                
            # Mean pooling of all face shots
            avg_emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm
                
            self.db[name] = {
                "embedding": avg_emb,
                "gaze_look_count": 0,
                "gaze_duration": 0.0,
                "is_currently_looking": False,
                "dwell_times": {},
                "last_seen": time.time()
            }
            print(f"[INFO] Enrolled Employee: {name} (using {len(embs)} reference shots)")
            count += 1
            
        if count == 0:
            print("[WARNING] No enrollment photos found. System will not recognize anyone! Please add photos to the 'enrollment' folder.")

    def extract_embedding(self, img):
        try:
            # 1. Robustly detect and crop the face using MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.face_detector.process(img_rgb)
            
            face_img = img
            if res.detections:
                # Get the most confident face
                det = max(res.detections, key=lambda x: x.score[0])
                bbox = det.location_data.relative_bounding_box
                h, w, _ = img.shape
                
                fx = int(bbox.xmin * w)
                fy = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                
                # Add a 10% margin around the face
                margin_x = int(fw * 0.1)
                margin_y = int(fh * 0.1)
                
                x1 = max(0, fx - margin_x)
                y1 = max(0, fy - margin_y)
                x2 = min(w, fx + fw + margin_x)
                y2 = min(h, fy + fh + margin_y)
                
                if x2 > x1 and y2 > y1:
                    face_img = img[y1:y2, x1:x2]

            # 2. Resize to standard ArcFace input dimension
            resized = cv2.resize(face_img, (112, 112))
            
            # Normalize pixel values (InsightFace standard: (x - 127.5) / 128)
            img_norm = resized.astype(np.float32)
            img_norm = (img_norm - 127.5) / 128.0
            
            # Transpose HWC (OpenCV) to CHW (ONNX standard)
            img_norm = np.transpose(img_norm, (2, 0, 1))
            img_norm = np.expand_dims(img_norm, axis=0)

            # Extract 512D biometric vector
            embeddings = self.session.run(None, {self.input_name: img_norm})[0]
            
            # L2 Normalize vector for easy cosine similarity calculations
            norm = np.linalg.norm(embeddings[0])
            if norm == 0:
                return None
            return embeddings[0] / norm
        except Exception:
            return None

    def identify_person_by_embedding(self, face_crop):
        """Compares target vector with database. Thread-safe."""
        embedding = self.extract_embedding(face_crop)
        if embedding is None:
            return "Unknown", False

        # Acquire lock to perform database query and updates
        with self.db_lock:
            best_match_id = None
            best_similarity = -1.0

            for pid, profile in self.db.items():
                db_emb = profile["embedding"]
                similarity = float(np.dot(embedding, db_emb))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = pid

            if best_similarity >= self.similarity_threshold:
                self.db[best_match_id]["last_seen"] = time.time()
                return best_match_id, False
            else:
                return "Unknown", False

    def log_dwell_time(self, profile_id, zone_name, elapsed_sec):
        """Accumulates dwell duration based on spatial overlap. Thread-safe."""
        with self.db_lock:
            if profile_id in self.db:
                dwell_dict = self.db[profile_id]["dwell_times"]
                if zone_name not in dwell_dict:
                    dwell_dict[zone_name] = 0.0
                dwell_dict[zone_name] += elapsed_sec

    def log_gaze_event(self, profile_id, is_looking_now, dt):
        """Processes gaze look transitions and tracks duration of active looks. Thread-safe."""
        with self.db_lock:
            if profile_id in self.db:
                profile = self.db[profile_id]
                # Trigger count increment strictly on False -> True transition
                if is_looking_now and not profile["is_currently_looking"]:
                    profile["gaze_look_count"] += 1
                
                # Accumulate exact time spent looking directly at camera
                if is_looking_now:
                    profile["gaze_duration"] += dt
                    
                profile["is_currently_looking"] = is_looking_now

    def get_metrics(self, profile_id, zone_name):
        """Retrieves statistics securely for GUI overlay. Thread-safe."""
        with self.db_lock:
            if profile_id in self.db:
                accum_dwell = self.db[profile_id]["dwell_times"].get(zone_name, 0.0) if zone_name else 0.0
                gaze_looks = self.db[profile_id]["gaze_look_count"]
                return accum_dwell, gaze_looks
            return 0.0, 0

    def export_logs(self, output_file="analytics_report.csv"):
        """Saves current session analytics to a CSV file in a per-employee format."""
        print(f"\n[INFO] Saving per-employee analytics to {output_file}...")
        with self.db_lock:
            # Collect all unique zones seen to create dynamic columns
            all_zones = set()
            for profile_id, data in self.db.items():
                all_zones.update(data["dwell_times"].keys())
            
            sorted_zones = sorted(list(all_zones))
            headers = ["Employee ID (Name_No)", "Total Camera Looks", "Time Looking At Camera (s)"]
            for z in sorted_zones:
                headers.append(f"{z} Dwell (s)")

            with open(output_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                
                for profile_id, data in self.db.items():
                    row = [
                        profile_id, 
                        data["gaze_look_count"], 
                        round(data.get("gaze_duration", 0.0), 1)
                    ]
                    for z in sorted_zones:
                        row.append(round(data["dwell_times"].get(z, 0.0), 1))
                    writer.writerow(row)
        print("[SUCCESS] Logs saved.")