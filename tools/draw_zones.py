# tools/draw_zones.py
import cv2
import json
import numpy as np
import sys
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ZONES_JSON_PATH, VIDEO_SOURCE

class ZoneDrawerUI:
    def __init__(self):
        self.current_points = []
        self.saved_zones = {}
        self.REF_FRAME_PATH = os.path.join(os.path.dirname(ZONES_JSON_PATH), "reference_frame.jpg")
        self.state = "DRAWING"
        
        # Hide root tkinter window
        self.root = tk.Tk()
        self.root.withdraw()
        
        # UI Button rectangles (x1, y1, x2, y2)
        self.btn_name = (20, 10, 180, 50)
        self.btn_clear = (200, 10, 360, 50)
        self.btn_save = (380, 10, 580, 50)

    def draw_button(self, img, text, rect, base_color):
        x1, y1, x2, y2 = rect
        # Sleek dark background with colored outline (Ghost button)
        cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 30), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), base_color, 1)
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_color, 1, cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y <= 60:
                # Check button clicks in the top header bar
                if self.btn_name[0] <= x <= self.btn_name[2] and self.btn_name[1] <= y <= self.btn_name[3]:
                    self.state = "TRIGGER_NAME"
                elif self.btn_clear[0] <= x <= self.btn_clear[2] and self.btn_clear[1] <= y <= self.btn_clear[3]:
                    self.current_points = []
                elif self.btn_save[0] <= x <= self.btn_save[2] and self.btn_save[1] <= y <= self.btn_save[3]:
                    self.state = "TRIGGER_SAVE"
            else:
                self.current_points.append([x, y])

    def run(self):
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("Error", "Could not access video source.")
            return

        window_name = "Professional Zone Configuration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            if self.state == "TRIGGER_NAME":
                if len(self.current_points) >= 3:
                    zone_name = simpledialog.askstring("Zone Name", "Enter unique name for this zone:", parent=self.root)
                    if zone_name and zone_name.strip():
                        self.saved_zones[zone_name.strip()] = self.current_points
                        self.current_points = []
                else:
                    messagebox.showwarning("Warning", "A zone must have at least 3 points to form a polygon.")
                self.state = "DRAWING"
                continue
                
            elif self.state == "TRIGGER_SAVE":
                os.makedirs(os.path.dirname(ZONES_JSON_PATH), exist_ok=True)
                with open(ZONES_JSON_PATH, 'w') as f:
                    json.dump(self.saved_zones, f, indent=4)
                cv2.imwrite(self.REF_FRAME_PATH, frame)
                messagebox.showinfo("Success", f"Saved {len(self.saved_zones)} zones successfully!")
                break

            canvas = frame.copy()
            overlay = frame.copy()

            # Render existing saved zones
            for idx, (name, points) in enumerate(self.saved_zones.items()):
                pts = np.array(points, dtype=np.int32)
                color = (255, 100, 0) if idx % 2 == 0 else (0, 150, 255)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(canvas, [pts], True, color, 2)
                cv2.putText(canvas, name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Render points currently being drawn
            if len(self.current_points) > 0:
                for pt in self.current_points:
                    cv2.circle(canvas, tuple(pt), 5, (0, 0, 255), -1)
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(canvas, [pts], True, (0, 0, 255), 2)

            alpha = 0.35
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
            
            # Draw UI Header (Top Bar)
            cv2.rectangle(canvas, (0, 0), (1280, 60), (15, 15, 20), -1)
            cv2.line(canvas, (0, 60), (1280, 60), (100, 100, 100), 1)
            
            cv2.putText(canvas, "ZONE CONFIGURATION", (650, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
            self.draw_button(canvas, "Name Zone", self.btn_name, (0, 255, 0))
            self.draw_button(canvas, "Clear Points", self.btn_clear, (0, 0, 255))
            self.draw_button(canvas, "Save & Exit", self.btn_save, (0, 200, 255))

            cv2.imshow(window_name, canvas)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    ZoneDrawerUI().run()