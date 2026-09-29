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
    assert R.shape == (3, 3)
    
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

    return np.degrees([x, y, z])  # Convert to degrees


   

import cv2
import numpy as np
import math
#np.int=int
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

model_points = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -63.6, -12.5],      # Chin
        [43.3, 32.7, -26.0],      # Right eye   corner
        [-43.3, 32.7, -26.0],     # Left eye   corner
        [28.9, -28.9, -24.1],     # Right mouth
        [-28.9, -28.9, -24.1],    # Left mouth
        ])


app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection','landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(640, 640))

video=cv2.VideoCapture(0)
if (video.isOpened()==False):
    print("Camera issueeeeee\n")
    exit()
flag=0
mthresh=0.6
ethresh=0.19
frame_check=5



while  True:
    ret,frame=video.read()
    faces=app.get(frame)
    for i in faces:
        f=i.bbox.astype(int)
        k=i.kps.astype(int)
        #print(i.landmark_3d_68)
        #print(k)
        l=(i.landmark_2d_106).astype(int)  

        #for idx, point in enumerate(l):
            #cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            #cv2.putText(frame, str(idx), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            
        lle=l[93]
        lre=l[89]
        rle=l[39]
        rre=l[35]
        lupper=l[94]
        rupper=l[40]
        lpupil=l[88]
        rpupil=l[38]
        llower=l[87]
        rlower=l[33]
        nose=l[86]
        chin=l[0]

        x1=int(min(lle[0],lre[0]))
        x2=int(max(lle[0],lre[0]))
        y1 = int(min(lupper[1], llower[1]))
        y2 = int(max(lupper[1], llower[1]))

        if x2>x1 and y2>y1:
            eye_img = frame[y1:y2, x1:x2]  
            if eye_img.size == 0:
                continue
            eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("Eye.jpeg",eye_gray)
            eye_blur = cv2.GaussianBlur(eye_gray, (7, 7), 0)
            cv2.imwrite("Eyeblur.jpeg",eye_blur)
            threshhold, thresh = cv2.threshold(eye_blur, 30, 255, cv2.THRESH_BINARY_INV)
            print("Threshold=  ",threshhold)
            cv2.imwrite("Aftertrhes.jpeg",thresh)
        else:
            continue



        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        iris_center=None
        if contours:
            largest = max(contours, key=cv2.contourArea)   #finds the largest contour ie, pupil
            (x, y), radius = cv2.minEnclosingCircle(largest)  # returns center and the radius also finds the smallest circle enclosing the largest contour
            iris_center = np.array([x + x1, y + y1])  # add x1 and y1 to convert it back to the full frame coordinates
        

        if iris_center is None:
            continue


        

        image_points = np.array([l[86],   # Nose tip
        l[0],    # Chin
        l[35],   # Right eye corner (right corner)
        l[93],   # right eye  left corner
        l[65],   # Right mouth
        l[69],   # Left mouth
        ], dtype="double")
        for idx in [86, 0, 35, 39, 65, 69]:
            x, y = l[idx]
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
                ], dtype="double")

        
        dist_coeffs = np.zeros((4, 1))


        eye_center_2d = ((l[93] + l[89]) / 2).astype(np.float32)  # Left eye center (2D)
        

        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        

        # 3D eye origin in camera space
        eye_model_point = np.array([[-30.0, 30.0, -26.0]])  # adjust if needed
        eye_center_3d, _ = cv2.projectPoints(eye_model_point, rvec, tvec, camera_matrix, dist_coeffs)
        eye_center_2d = eye_center_3d[0][0]  # 2D point of eye center in image

# Gaze direction vector in 2D (image space)
        gaze_2d = np.array(iris_center) - eye_center_2d

# Convert to pseudo 3D direction in camera space
        gaze_vector = np.array([gaze_2d[0], gaze_2d[1], focal_length])  # Use focal_length as Z to approximate perspective
        gaze_vector /= np.linalg.norm(gaze_vector)

        


        


# Project a 3D point (100 mm in front of the nose tip) to show head direction
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rvec, tvec, camera_matrix, dist_coeffs)
        p1 = tuple(image_points[0].astype(int))  # Nose tip
        p2 = tuple(nose_end_point2D[0][0].astype(int))  # Projected point

        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        R, _ = cv2.Rodrigues(rvec)  # converts rotation vector to matrix
        pitch, yaw, roll = rotationMatrixToEulerAngles(R) # converts matrix to pitch , yaw, roll

        head_dir = R @ np.array([0, 0, 1]) 

        

        # Inverse of head rotation to move eye_vec into head frame
        #R_inv = R.T
        gaze_head_frame = R.T @ gaze_vector  # make sthe gaze dir relative to head pose and not the camera position

        gaze_line_end = (eye_center_2d + (gaze_vector[:2] * 50)).astype(int)
        cv2.line(frame, tuple(eye_center_2d.astype(int)), tuple(gaze_line_end), (0, 255, 255), 2)

        gaze_head_frame /= np.linalg.norm(gaze_head_frame)
        print("Gaze vector in head frame:", gaze_head_frame)


        #print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")
        if gaze_head_frame[0]> 0.2:
            cv2.putText(frame,"right",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        elif gaze_head_frame[0] < -0.2:
            cv2.putText(frame,"left",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        elif gaze_head_frame[1] > 0.2:
            cv2.putText(frame,"Down",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        elif gaze_head_frame[1] < -0.2:
            cv2.putText(frame,"Up",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        else:
            cv2.putText(frame,"Center",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        

        #for i, point in enumerate(l):
            #print(f"Point {i}: {point}")
            #cv2.putText(frame, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)



        #inner mouth points
        a=np.linalg.norm(l[62]-l[60])
        b=np.linalg.norm(l[69]-l[65])
        m=a/b
        #print("value m: ",m)
        if(m>0.3):
            cv2.putText(frame, "Yawning", (f[0]+10, f[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif (m>0.1):
            cv2.putText(frame, "Speaking", (f[0]+10, f[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        

        
        r=left_ear(l)
        le=right_ear(l)
        ear=(r+le)/2.0
        #print("Eye valueeee:",ear)
        if(ear<ethresh):
            flag+=1
            if(flag>=frame_check):
                
                print("Alerrrrrzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
                cv2.putText(frame, "****************ALERT!****************", (f[0]+10, f[1]-120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
                cv2.putText(frame, "@@@@@@@@@@@DROWSY@@@@@@@@@@@@@", (f[0]+10, f[1]-100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
        else:
            flag=0
        


        

        
            
        




        #print(l)
        for m in l:
                     
            x=m[0]
            y=m[1]
            j=(x,y)
            #cv2.circle(frame,j,1,(0,255,0),1,cv2.LINE_AA)
        #for i in k:
            #cv2.circle(img,i,2,(0,0,255),2)
        cv2.rectangle(frame,(f[0],f[1]),(f[2],f[3]),(255,255,255),3)
    cv2.imshow("Frame=",frame)
    key=cv2.waitKey(1)
    if (key==ord('q')):
        break
        
cv2.destroyAllWindows()
video.release()













