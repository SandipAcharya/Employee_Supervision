# MediaPipe Hybrid Gaze (Head Pose + Iris)
#This script uses MediaPipe Face Mesh to calculate coarse head orientation first (PnP alignment), then checks if the pupil is centered horizontally inside the eye contour[2][7].

# core/gaze.py
import cv2
import numpy as np
import mediapipe as mp
from config.settings import MAX_YAW_DEGREE, MAX_PITCH_DEGREE, IRIS_CENTER_THRESHOLD

class HybridGazeEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,  # Enables the dedicated 468-477 iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_gaze(self, frame_rgb):
        """
        Processes a full RGB frame.
        Returns: Dict mapping face index to (is_looking, yaw, pitch)
        """
        h, w, _ = frame_rgb.shape
        results = self.face_mesh.process(frame_rgb)
        
        gaze_results = []
        if not results.multi_face_landmarks:
            return gaze_results

        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = face_landmarks.landmark
            
            # --- 1. Head Pose (Coarse Check) ---
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])

            image_points = np.array([
                (landmarks[4].x * w, landmarks[4].y * h),     # Nose tip
                (landmarks[152].x * w, landmarks[152].y * h), # Chin
                (landmarks[33].x * w, landmarks[33].y * h),   # Left eye left corner
                (landmarks[263].x * w, landmarks[263].y * h), # Right eye right corner
                (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth corner
                (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth corner
            ], dtype="double")

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1))
            
            _, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            rmat, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rmat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
            
            pitch = euler_angles[0, 0]
            yaw = euler_angles[1, 0]

            # Normalize OpenCV PnP flips (often returns ~180 instead of ~0)
            if pitch > 90: pitch -= 180
            elif pitch < -90: pitch += 180
            
            if yaw > 90: yaw -= 180
            elif yaw < -90: yaw += 180
            
            # Sometimes pitch needs a sign flip depending on camera mapping, but normalization handles the jump.
            left_eye_outer = landmarks[130]
            left_eye_inner = landmarks[133]
            left_iris = landmarks[468]  # Center index of iris array

            # Horizontal centering ratio
            dist_outer_to_iris = np.linalg.norm(np.array([left_iris.x - left_eye_outer.x, left_iris.y - left_eye_outer.y]))
            dist_outer_to_inner = np.linalg.norm(np.array([left_eye_inner.x - left_eye_outer.x, left_eye_inner.y - left_eye_outer.y]))
            
            if dist_outer_to_inner == 0:
                continue

            iris_ratio = dist_outer_to_iris / dist_outer_to_inner
            pupil_offset = abs(iris_ratio - 0.5)

            # --- 3. Verdict ---
            is_head_directed = (abs(yaw) <= MAX_YAW_DEGREE) and (abs(pitch) <= MAX_PITCH_DEGREE)
            is_iris_centered = pupil_offset <= IRIS_CENTER_THRESHOLD
            is_looking = bool(is_head_directed and is_iris_centered)

            gaze_results.append({
                "is_looking": is_looking,
                "yaw": yaw,
                "pitch": pitch,
                "nose_coords": (int(landmarks[4].x * w), int(landmarks[4].y * h))
            })

        return gaze_results