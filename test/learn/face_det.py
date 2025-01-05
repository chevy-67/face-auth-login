import cv2

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

img = cv2.imread('assets/ref.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face = facedetect.detectMultiScale(gray,1.3,5)

for(x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('r',img)
k = cv2.waitKey()
if(k==ord('q')):
    cv2.destroyAllWindows()