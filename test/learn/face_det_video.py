import cv2

sample = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

vid = cv2.VideoCapture(0)

while True:
    ret,img = vid.read()
    face = sample.detectMultiScale(img,1.4,5)
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('video',img)
    k = cv2.waitKey(1)
    if(k==27):
        break

cv2.destroyAllWindows()
