import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from tkinter import *
import threading

def left_ear(eye):
    A = np.linalg.norm(eye[95] - eye[90])
    B = np.linalg.norm(eye[96] - eye[91])
    C = np.linalg.norm(eye[89] - eye[93])
    return (A + B) / (2.0 * C)

def right_ear(eye):
    A = np.linalg.norm(eye[41] - eye[36])
    B = np.linalg.norm(eye[42] - eye[37])
    C = np.linalg.norm(eye[35] - eye[39])
    return (A + B) / (2.0 * C)

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])   # Roll
        y = np.arctan2(-R[2, 0], sy)       # Pitch
        z = np.arctan2(R[1, 0], R[0, 0])   # Yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])

def get_eye_gaze(eye_pts, frame):
    x1 = int(min(eye_pts[0][0], eye_pts[1][0]))
    x2 = int(max(eye_pts[0][0], eye_pts[1][0]))
    y1 = int(min(eye_pts[2][1], eye_pts[3][1]))
    y2 = int(max(eye_pts[2][1], eye_pts[3][1]))

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    eye_img = frame[y1:y2, x1:x2]
    if eye_img.size == 0:
        return None, None, None

    eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_blur = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    _, thresh = cv2.threshold(eye_blur, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    (x, y), _ = cv2.minEnclosingCircle(largest)
    iris_center = np.array([x, y])
    eye_width = x2 - x1
    eye_height = y2 - y1
    
    if eye_width == 0 or eye_height == 0: # Safety catch division by zero
        return None, None, None
        
    norm_x = (iris_center[0] - eye_width / 2) / (eye_width / 2)
    norm_y = (iris_center[1] - eye_height / 2) / (eye_height / 2)
    return (norm_x, norm_y), (x1, y1), iris_center

def draw_gaze(frame, gaze, origin, iris_center, label):
    if gaze is None:
        return
    norm_x, norm_y = gaze
    direction = "Center"
    if norm_x > 0.3:
        direction = "Left"
    elif norm_x < -0.3:
        direction = "Right"
    elif norm_y > 0.35:
        direction = "Down"
    elif norm_y < -0.35:
        direction = "Up"
    cv2.putText(frame, f"{label}: {direction}", (origin[0], origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.circle(frame, (int(origin[0] + iris_center[0]), int(origin[1] + iris_center[1])), 2, (0, 255, 0), -1)


class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Face Tracking System")
        self.root.geometry("300x150")
        
        # UI Elements
        Label(self.root, text="Drowsiness & Face Tracking", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        self.btn_start = Button(self.root, text="Start Detection", width=15, command=self.start_detection, bg="lightgreen")
        self.btn_start.pack(pady=5)
        
        self.btn_stop = Button(self.root, text="Stop Detection", width=15, command=self.stop_detection, bg="salmon")
        self.btn_stop.pack(pady=5)
        
        self.running = False
        self.thread = None
        
    def start_detection(self):
        if not self.running:
            self.running = True
            # We run the camera processing in a separate thread so Tkinter UI doesn't freeze
            self.thread = threading.Thread(target=self.camera_loop)
            self.thread.daemon = True # Will exit automatically if the main Tkinter window closes
            self.thread.start()
            
    def stop_detection(self):
        self.running = False

    def camera_loop(self):
        try:
            print("Waking up FaceAnalysis Model...")
            app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection','landmark_2d_106'])
            app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"Error loading model: {e}")
            self.running = False
            return
            
        model_points = np.array([
            [0.0, 0.0, 0.0], [0.0, -63.6, -12.5], [43.3, 32.7, -26.0], 
            [-43.3, 32.7, -26.0], [28.9, -28.9, -24.1], [-28.9, -28.9, -24.1]
        ])

        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Camera error! Please ensure it's not being used by another application.")
            self.running = False
            return

        flag = 0
        ethresh = 0.19
        frame_check = 5
        blink = 0
        yawn = 0

        while self.running:
            ret, frame = video.read()
            if not ret:
                break
                
            faces = app.get(frame)
            
            # Feature 1: Robust Filter to logic (Find the single largest face)
            if len(faces) > 1:
                faces = [max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))]

            for i in faces:
                f = i.bbox.astype(int)
                l = i.landmark_2d_106.astype(int)

                # EYE LANDMARKS + EYE GAZE
                left_eye_pts = [l[93], l[89], l[94], l[87]]
                right_eye_pts = [l[39], l[35], l[40], l[33]]
                left_gaze, left_origin, left_center = get_eye_gaze(left_eye_pts, frame)
                right_gaze, right_origin, right_center = get_eye_gaze(right_eye_pts, frame)
                draw_gaze(frame, left_gaze, left_origin, left_center, "Left Eye")
                draw_gaze(frame, right_gaze, right_origin, right_center, "Right Eye")

                # HEAD POSE ESTIMATION (Nose vector line calculation)
                image_points = np.array([l[86], l[0], l[35], l[93], l[65], l[69]], dtype="double")
                focal_length = frame.shape[1]
                center = (frame.shape[1] / 2, frame.shape[0] / 2)
                camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
                dist_coeffs = np.zeros((4, 1))
                success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
                
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    pitch, yaw_angle, roll = rotationMatrixToEulerAngles(R)
                    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rvec, tvec, camera_matrix, dist_coeffs)
                    cv2.line(frame, tuple(image_points[0].astype(int)), tuple(nose_end_point2D[0][0].astype(int)), (255, 0, 0), 2)

                # MOUTH LOGIC (Speaking & Yawning)
                a = np.linalg.norm(l[62] - l[60])
                b = np.linalg.norm(l[69] - l[65])
                
                if b > 0:
                    m = a / b
                    if m > 0.3:
                        yawn += 1
                        cv2.putText(frame, "Yawning", (f[0]+10, f[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif m > 0.1:
                        cv2.putText(frame, "Speaking", (f[0]+10, f[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # EYE ASPECT RATIO SCORING (Drowsiness / Blinks)
                r = left_ear(l)
                le = right_ear(l)
                ear = (r + le) / 2.0
                
                if ear < ethresh:
                    flag += 1
                    if flag >= frame_check:
                        blink += 1 # we append drop frames to act as blink counters
                        cv2.putText(frame, "****************ALERT!****************", (f[0]+10, f[1]-120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "@@@@@@@@@@@DROWSY@@@@@@@@@@@@@", (f[0]+10, f[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    flag = 0

                # GENERAL OVERLAYS & UI
                cv2.putText(frame, f"Blinks / Eye Drops: {blink}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawns: {yawn}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                cv2.putText(frame, "Advanced Tracking Mode", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
                cv2.rectangle(frame, (f[0], f[1]), (f[2], f[3]), (255, 255, 255), 2)

            cv2.imshow("Detection Frame - Press q to close", frame)
            
            # Allow clean breakout if running=False or 'q' pushed on cv2 window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    app = FaceApp(root)
    root.mainloop()
