import cv2
import numpy as np
from time import sleep

def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

class Solution:
    def forward(self, video_path: str) -> int:
        min_width=80
        min_height=80
        offset=6
        delay=60
        
        detections=[]
        vehicles=0
        tracked_objects={}
        missed_frames={}
        next_id=0
        line_pos=None
        
        cap = cv2.VideoCapture(video_path)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if line_pos is None:
                h = frame.shape[0]
                line_pos = h // 2 - 50
            
            tempo = float(1/delay)
            sleep(tempo) 
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey = cv2.bilateralFilter(grey, 9, 75, 75)
            blur = cv2.GaussianBlur(grey, (3, 3), 5)
            fg_mask = bg_subtractor.apply(blur)
            dilated = cv2.dilate(fg_mask, np.ones((5, 5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for(i, c) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                validate_contour = (w >= min_width) and (h >= min_height) and (area >= 800)
                if not validate_contour:
                    continue

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)        
                center = get_center(x, y, w, h)
                detections.append(center)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

            used = set()
            for cx, cy in detections:
                best_dist = 200
                best_id = -1
                
                for obj_id in tracked_objects.keys():
                    if obj_id in used:
                        continue
                    prev_x, prev_y = tracked_objects[obj_id]
                    dist = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_id = obj_id
                
                if best_id >= 0:
                    prev_x, prev_y = tracked_objects[best_id]
                    if (prev_y < line_pos and cy > line_pos) or (prev_y > line_pos and cy < line_pos):
                        vehicles += 1
                        print("car is detected: " + str(vehicles))
                    tracked_objects[best_id] = (cx, cy)
                    missed_frames[best_id] = 0
                    used.add(best_id)
                else:
                    tracked_objects[next_id] = (cx, cy)
                    missed_frames[next_id] = 0
                    next_id += 1
            
            for oid in list(tracked_objects.keys()):
                if oid not in used:
                    missed_frames[oid] = missed_frames.get(oid, 0) + 1
                    if missed_frames[oid] > 5:
                        del tracked_objects[oid]
                        del missed_frames[oid]
            
            w = frame.shape[1]
            cv2.line(frame, (0, line_pos), (w, line_pos), (255, 0, 0), 2)
            
            cv2.imshow("Vehicle Counter", frame)
            cv2.imshow("Foreground Mask", fg_mask)

            if cv2.waitKey(1) == 27:
                break
        
        cv2.destroyAllWindows()
        cap.release()
        
        return vehicles

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'vehant_hackathon_video_10.mp4'
    
    sol = Solution()
    count = sol.forward(video_path)
    print(count)