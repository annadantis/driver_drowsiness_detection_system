


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


   

import cv2
import numpy as np
import math
#np.int=int
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image



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
        x1 = min(lle[0], lre[0])
        x2 = max(lle[0], lre[0])
        y1 = min(lupper[1], llower[1])
        y2 = max(lupper[1], llower[1])   

        p=np.linalg.norm(lle-lre)
        ed=4
        d=65
        focal=p*d/ed
        #print("Focal length=  ",focal)

        #print("Nose tip=   ",l[86])
        #print("right eye centre:  ",l[38])
        #print("left eye cen:   ",l[88])



        a1=np.linalg.norm(l[86]-l[38])
        b1=np.linalg.norm(l[86]-l[88])

        #print("nose and right:  ",a1)
        #print("nose and left:   ",b1)

        for i, point in enumerate(l):
            #print(f"Point {i}: {point}")
            cv2.putText(frame, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)



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
        


        lew=np.linalg.norm(lre-lle)
        lho=np.linalg.norm(lpupil-lle)
        lhr=lho/lew
        #print("Left hori ratio=== :  ",lhr)

        rew=np.linalg.norm(rre-rle)
        rho=np.linalg.norm(rpupil-rle)
        rhr=rho/rew
        #print("right hori ratio=== :  ",rhr)

        x=(lhr+rhr)/2
        #print("Hoizn= ",x) 



        lheight=np.linalg.norm(llower-lupper)
        lvo=np.linalg.norm(lpupil-lupper)
        lvr=lvo/lheight
        print("Left verti ratio ===",lvr)

        rheight=np.linalg.norm(rlower-rupper)
        rvo=np.linalg.norm(rpupil-rupper)
        rvr=rvo/rheight
        print("Left verti ratio ===",rvr)

        y=(lvr+rvr)/2

        print("VErti= ",y)

        if(x<0.49):
            cv2.putText(frame,"left",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        elif (x>0.51):
            cv2.putText(frame,"right",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        elif (y<0.5):
            cv2.putText(frame,"Up",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        elif (y>0.5):
            cv2.putText(frame,"Down",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        else:
            cv2.putText(frame,"Center",(f[0]+10,f[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)




        #print(l)
        for m in l:
                     
            x=m[0]
            y=m[1]
            j=(x,y)
            cv2.circle(frame,j,1,(0,255,0),1,cv2.LINE_AA)
        #for i in k:
            #cv2.circle(img,i,2,(0,0,255),2)
        cv2.rectangle(frame,(f[0],f[1]),(f[2],f[3]),(255,255,255),3)
    cv2.imshow("Frame=",frame)
    key=cv2.waitKey(1)
    if (key==ord('q')):
        break
        
cv2.destroyAllWindows()
video.release()












