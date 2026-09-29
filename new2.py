import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from playsound import playsound

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




def get_eye_gaze(eye_pts, frame):
    x1 = int(min(eye_pts[0][0], eye_pts[1][0]))
    x2 = int(max(eye_pts[0][0], eye_pts[1][0]))
    y1 = int(min(eye_pts[2][1], eye_pts[3][1]))
    y2 = int(max(eye_pts[2][1], eye_pts[3][1]))

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    eye_img = frame[y1:y2, x1:x2]  # crop eye region from the frame
    if eye_img.size == 0:
        return None, None, None

    eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_blur = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    _, thresh = cv2.threshold(eye_blur, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea) # finds the largest contour on the basis of area
    (x, y), _ = cv2.minEnclosingCircle(largest) # finds the min circle enclosing the largest contour
    iris_center = np.array([x, y]) 
    eye_width = x2 - x1
    eye_height = y2 - y1
    norm_x = (iris_center[0] - eye_width / 2) / (eye_width / 2)
    norm_y = (iris_center[1] - eye_height / 2) / (eye_height / 2)
    print("X==",norm_x,"\nY==",norm_y)
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

# Model setup
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection','landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(640, 640))

model_points = np.array([
    [0.0, 0.0, 0.0], [0.0, -63.6, -12.5], [43.3, 32.7, -26.0], [-43.3, 32.7, -26.0], [28.9, -28.9, -24.1], [-28.9, -28.9, -24.1]
])

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Camera issue!")
    exit()

flag = 0
mthresh = 0.6
ethresh = 0.19
frame_check = 5
blink=0
yawn=0
f=0
while True:
    ret, frame = video.read()
    if not ret:
        break
    faces = app.get(frame)
    if len(faces) > 1:
        faces = [max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))]

    for i in faces:
        f = i.bbox.astype(int)
        l = i.landmark_2d_106.astype(int)

        # Eye landmark setup
        left_eye_pts = [l[93], l[89], l[94], l[87]]
        right_eye_pts = [l[39], l[35], l[40], l[33]]
        left_gaze, left_origin, left_center = get_eye_gaze(left_eye_pts, frame)
        right_gaze, right_origin, right_center = get_eye_gaze(right_eye_pts, frame)
        draw_gaze(frame, left_gaze, left_origin, left_center, "Left Eye")
        draw_gaze(frame, right_gaze, right_origin, right_center, "Right Eye")

        
        # EAR and mouth logic
        a = np.linalg.norm(l[62] - l[60])
        b = np.linalg.norm(l[69] - l[65])
        m = a / b
        if m > 0.3:
            yawn+=1
            
            cv2.putText(frame, "Yawning", (f[0]+10, f[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif m > 0.1:
            cv2.putText(frame, "Speaking", (f[0]+10, f[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 

        r = left_ear(l)
        le = right_ear(l)
        ear = (r + le) / 2.0
        if ear < ethresh:
            flag += 1
            
            if flag >= frame_check:
                blink=blink+1
                cv2.putText(frame, "****************ALERT!****************", (f[0]+10, f[1]-120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "@@@@@@@@@@@DROWSY@@@@@@@@@@@@@", (f[0]+10, f[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0
        
        
        
        cv2.putText(frame, f"Blinks: {blink}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawns: {yawn}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Face box
        cv2.rectangle(frame, (f[0], f[1]), (f[2], f[3]), (255, 255, 255), 2)
        cv2.putText(frame, "Drowsiness Detection", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()