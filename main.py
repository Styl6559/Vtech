import cv2
import numpy as np
from time import sleep

def get_center(x, y, w, h):
    return x + w // 2, y + h // 2

class Solution:
    def forward(self, video_path: str) -> int:
        min_width = 30
        min_height = 30
        delay = 60
        LINE_TOLERANCE = 12

        vehicles = 0
        tracked_objects = {}
        next_id = 0
        line_pos = None

        cap = cv2.VideoCapture(video_path)
        bg = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=50,
            detectShadows=False
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if line_pos is None:
                h = frame.shape[0]
                line_pos = h // 2 - 50

            sleep(1 / delay)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            fg = bg.apply(gray)
            fg[fg > 0] = 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)

            contours, _ = cv2.findContours(
                fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            detections = []

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = cv2.contourArea(c)

                if w < min_width or h < min_height or area < 300:
                    continue

                ar = w / float(h)
                if ar < 0.2 or ar > 3.0:
                    continue

                cx, cy = get_center(x, y, w, h)
                detections.append((cx, cy))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            used = set()

            for cx, cy in detections:
                best_dist = 80
                best_id = -1

                for oid, (px, py, crossed, miss) in tracked_objects.items():
                    if oid in used:
                        continue
                    dist = np.hypot(cx - px, cy - py)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = oid

                if best_id != -1:
                    px, py, crossed, miss = tracked_objects[best_id]

                    if not crossed and abs(cy - line_pos) <= LINE_TOLERANCE:
                        vehicles += 1
                        crossed = True
                        print("vehicle counted:", vehicles)

                    tracked_objects[best_id] = (cx, cy, crossed, 0)
                    used.add(best_id)
                else:
                    tracked_objects[next_id] = (cx, cy, False, 0)
                    next_id += 1

            for oid in list(tracked_objects.keys()):
                if oid not in used:
                    px, py, crossed, miss = tracked_objects[oid]
                    miss += 1
                    if miss > 8:
                        del tracked_objects[oid]
                    else:
                        tracked_objects[oid] = (px, py, crossed, miss)

            w = frame.shape[1]
            cv2.line(frame, (0, line_pos), (w, line_pos), (255, 0, 0), 2)

            cv2.imshow("Vehicle Counter", frame)
            cv2.imshow("Foreground Mask", fg)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return vehicles

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "vehant_hackathon_video_10.mp4"
    print(Solution().forward(video_path))
