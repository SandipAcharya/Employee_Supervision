# Point-in-polygon zone checker
# This script loads the static JSON coordinates and performs point-in-polygon math using cv2.pointPolygonTest to see if a tracked person is standing or sitting inside a defined zone[3][5].

# core/zones.py
# core/zones.py
import json
import cv2
import numpy as np
import os
from config.settings import ZONES_JSON_PATH

class ZoneChecker:
    def __init__(self):
        self.raw_zones = {}
        self.aligned_zones = {}
        self.ref_frame = None
        self.load_zones()
        
        # Load reference image
        ref_path = os.path.join(os.path.dirname(ZONES_JSON_PATH), "reference_frame.jpg")
        if os.path.exists(ref_path):
            self.ref_frame = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            
        # Initialize ORB detector for image registration
        self.orb = cv2.ORB_create(nfeatures=1000)

    def load_zones(self):
        if not os.path.exists(ZONES_JSON_PATH):
            return
        with open(ZONES_JSON_PATH, 'r') as f:
            data = json.load(f)
            for name, pts in data.items():
                self.raw_zones[name] = np.array(pts, dtype=np.int32)
        # Initially, aligned zones match raw zones
        self.aligned_zones = self.raw_zones.copy()

    def align_zones_to_current_view(self, current_frame):
        """
        Calculates homography between reference frame and current frame.
        Warps the zone boundaries to match if the camera was bumped or shifted.
        """
        if self.ref_frame is None or len(self.raw_zones) == 0:
            return

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # 1. Find keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.ref_frame, None)
        kp2, des2 = self.orb.detectAndCompute(current_gray, None)

        if des1 is None or des2 is None:
            return

        # 2. Match descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # 3. Use top matches to estimate perspective matrix (Homography)
        if len(matches) > 15:  # Require minimum point pairings
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # RANSAC isolates true matches from dynamic background noise (like walking people)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                new_aligned = {}
                for name, pts in self.raw_zones.items():
                    # Format coordinates to transform
                    pts_reshaped = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                    # Warp the coordinates using Homography transformation matrix
                    warped_pts = cv2.perspectiveTransform(pts_reshaped, H)
                    new_aligned[name] = np.array(warped_pts.reshape(-1, 2), dtype=np.int32)
                
                self.aligned_zones = new_aligned

    def check_position(self, point):
        """Checks if coordinate falls inside warped polygon."""
        px, py = int(point[0]), int(point[1])
        for name, poly in self.aligned_zones.items():
            dist = cv2.pointPolygonTest(poly, (px, py), False)
            if dist >= 0:
                return name
        return None