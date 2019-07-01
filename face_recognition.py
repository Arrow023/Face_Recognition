'''This code is to obtain sample images from the camera feed present in the computer for face recognition.
These images will be used to train the model for face recognition'''
import cv2
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cropped_faces= img[y:y+h,x:x+w]
    return cropped_faces
cap=cv2.VideoCapture(0)
count=0
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name_path='D:/My Files/Face Recognition/faces/piyush'+str(count)+'.jpg'
        '''Note that there should be the folder named 'faces' as in the specified path, so it carefully'''
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face cropper',face)
    if cv2.waitKey(1) & 0xFF == ord('l'):
        break
cap.release()
cv2.destroyAllWindows()
print('Collecting samples complete...')


