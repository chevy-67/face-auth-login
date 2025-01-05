import cv2
import face_recognition as fr

ref_img = fr.load_image_file('assets/ref1.jpg')
ref_encode = fr.face_encodings(ref_img)

if(len(ref_encode)==0):
    print("No face detected in reference:(")
    exit()

ref_face = ref_encode[0]

vid = cv2.VideoCapture(0)

while True:
    ret,frame = vid.read()
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame,face_locations)

    for face_encode,face_location in zip(face_encodings,face_locations):
        matches = fr.compare_faces([ref_face],face_encode)
        face_dist = fr.face_distance([ref_face],face_encode)[0]

        max_threshold = 0.6
        if(matches[0] and face_dist<max_threshold):
            color = (0,255,0)
            label = 'Match'
        else:
            color = (0,0,255)
            label = 'Not Matched'
        
        top,right,bottom,left = face_location
        cv2.rectangle(frame,(left,top),(right,bottom),color,2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow('Frame',frame)
    k = cv2.waitKey(1)
    if(k==27):
        break

vid.release()
cv2.destroyAllWindows()