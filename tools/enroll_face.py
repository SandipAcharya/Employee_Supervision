# tools/enroll_face.py
import cv2
import os
import glob
import tkinter as tk
from tkinter import simpledialog, messagebox
import mediapipe as mp

class EnrollmentUI:
    def __init__(self):
        self.state = "IDLE"
        self.emp_name = None
        self.shots_taken = 0
        self.required_shots = ["Center", "Left", "Right"]
        self.captured_faces = [] # List of tuples: (image, bbox)
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.enrollment_dir = os.path.join(self.base_dir, "enrollment")
        os.makedirs(self.enrollment_dir, exist_ok=True)
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        
        self.root = tk.Tk()
        self.root.withdraw()
        
        # UI Rects
        self.btn_start = (300, 620, 580, 680)
        self.btn_exit = (700, 620, 980, 680)
        self.btn_capture = (500, 620, 780, 680)

    def draw_button(self, img, text, rect, base_color):
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 30), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), base_color, 1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 1, cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "IDLE":
                if self.btn_start[0] <= x <= self.btn_start[2] and self.btn_start[1] <= y <= self.btn_start[3]:
                    name = simpledialog.askstring("Employee Profile", "Enter New Employee Name:", parent=self.root)
                    if name and name.strip():
                        self.emp_name = name.strip().replace(" ", "_")
                        self.state = "CAPTURE_MODE"
                        self.shots_taken = 0
                        self.captured_faces = []
                elif self.btn_exit[0] <= x <= self.btn_exit[2] and self.btn_exit[1] <= y <= self.btn_exit[3]:
                    self.state = "EXIT"
            
            elif self.state == "CAPTURE_MODE":
                if self.btn_capture[0] <= x <= self.btn_capture[2] and self.btn_capture[1] <= y <= self.btn_capture[3]:
                    self.state = "TRIGGER_CAPTURE"

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        window_name = "Pro Enrollment Wizard"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret: break

            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Dark bottom bar
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 600), (w, h), (15, 15, 20), -1)
            cv2.line(overlay, (0, 600), (w, 600), (100, 100, 100), 1)
            cv2.addWeighted(overlay, 0.9, display_frame, 0.1, 0, display_frame)

            if self.state == "IDLE":
                cv2.putText(display_frame, "CREATE NEW BIOMETRIC PROFILE", (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                self.draw_button(display_frame, "START ENROLLMENT", self.btn_start, (0, 255, 0))
                self.draw_button(display_frame, "EXIT", self.btn_exit, (0, 0, 255))
                
            elif self.state == "EXIT":
                break

            elif self.state == "CAPTURE_MODE" or self.state == "TRIGGER_CAPTURE":
                # Face Detection
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.face_detector.process(img_rgb)
                
                captured_face_box = None
                if res.detections:
                    det = max(res.detections, key=lambda x: x.score[0])
                    bbox = det.location_data.relative_bounding_box
                    fx, fy = int(bbox.xmin * w), int(bbox.ymin * h)
                    fw, fh = int(bbox.width * w), int(bbox.height * h)
                    captured_face_box = (fx, fy, fw, fh)
                    
                    # Target Brackets
                    l = 25; t = 2; color = (0, 255, 255)
                    cv2.line(display_frame, (fx, fy), (fx + l, fy), color, t); cv2.line(display_frame, (fx, fy), (fx, fy + l), color, t)
                    cv2.line(display_frame, (fx+fw, fy), (fx+fw - l, fy), color, t); cv2.line(display_frame, (fx+fw, fy), (fx+fw, fy + l), color, t)
                    cv2.line(display_frame, (fx, fy+fh), (fx + l, fy+fh), color, t); cv2.line(display_frame, (fx, fy+fh), (fx, fy+fh - l), color, t)
                    cv2.line(display_frame, (fx+fw, fy+fh), (fx+fw - l, fy+fh), color, t); cv2.line(display_frame, (fx+fw, fy+fh), (fx+fw, fy+fh - l), color, t)
                
                # Instructions Overlay
                instruction = self.required_shots[self.shots_taken]
                cv2.putText(display_frame, f"Profile: {self.emp_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(display_frame, f"Step {self.shots_taken+1}/3: Look {instruction}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                
                self.draw_button(display_frame, f"CAPTURE ({instruction})", self.btn_capture, (0, 200, 255))
                
                if self.state == "TRIGGER_CAPTURE":
                    if captured_face_box:
                        self.captured_faces.append((frame.copy(), captured_face_box))
                        self.shots_taken += 1
                        
                        if self.shots_taken >= 3:
                            self.state = "SAVING"
                        else:
                            self.state = "CAPTURE_MODE"
                    else:
                        messagebox.showwarning("Warning", "No face detected! Please ensure you are visible.")
                        self.state = "CAPTURE_MODE"

            elif self.state == "SAVING":
                cv2.putText(display_frame, "Saving Multi-Shot Profile...", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(100) # Give UI time to update
                
                # Create a subfolder for this employee for multi-shot averaging
                person_dir = os.path.join(self.enrollment_dir, self.emp_name)
                os.makedirs(person_dir, exist_ok=True)
                
                for idx, (img, bbox) in enumerate(self.captured_faces):
                    fx, fy, fw, fh = bbox
                    margin_x, margin_y = int(fw * 0.5), int(fh * 0.5)
                    x1 = max(0, fx - margin_x)
                    y1 = max(0, fy - margin_y)
                    x2 = min(w, fx + fw + margin_x)
                    y2 = min(h, fy + fh + margin_y)
                    
                    face_crop = img[y1:y2, x1:x2]
                    filepath = os.path.join(person_dir, f"{self.emp_name}_{self.required_shots[idx].lower()}.jpg")
                    cv2.imwrite(filepath, face_crop)
                    
                messagebox.showinfo("Success", f"Master Template for '{self.emp_name}' successfully created!")
                self.state = "IDLE"

            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    EnrollmentUI().run()
