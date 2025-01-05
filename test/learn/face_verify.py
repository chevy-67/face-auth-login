import cv2

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

ref_img_path = 'assets/ref.jpg'
ref_img = cv2.imread(ref_img_path)
ref_gray = cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)

ref_face = facedetect.detectMultiScale(ref_gray,1.3,4)

(x,y,w,h) = ref_face[0]
ref_face = ref_gray[y:y+h,x:x+w]

vid = cv2.VideoCapture(0)

while True:
    ret,frame = vid.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    live_face = facedetect.detectMultiScale(frame_gray,1.3,4)

    for(x,y,w,h) in live_face:
        face = frame_gray[y:y+h,x:x+w]
        resized_ref = cv2.resize(ref_face,(w,h))

        difference = cv2.absdiff(face,resized_ref)
        score = (difference**2).mean()

        if(score<100):
            label = "Matched"
            color = (0,255,0)
        else:
            label = "Not Matched"
            color = (0,0,255)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imshow('Face Auth',frame)

    k = cv2.waitKey(1)
    if(k==27):
        break
vid.release()
cv2.destroyAllWindows()


