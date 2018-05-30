import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

face_id = input('\nEnter user id:  ')
print("\n[INFO] Initializing face capture. Look the camera and wait ...")

count = 0
capture = False


while(True):
  ret, img = cam.read()
  img = cv2.flip(img, 1)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_detector.detectMultiScale(gray, 1.3, 5)

  for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)     
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_detector.detectMultiScale(roi_gray, 1.3, 5)
    for (ex,ey,ew,eh) in eyes:
      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),1)

    cv2.putText(img, "Press 'c' to capture!",  (x+5,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(img, "id: {}".format(face_id),   (x+5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, "dataset: {}/30".format(count),   (x+5, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

  if capture:
    count += 1
    path = "dataset/user." + str(face_id) + '.' + str(count) + ".jpg"
    cv2.imwrite(path, gray[y:y+h,x:x+w])
    print("saving: " + path)

  cv2.imshow('image', img)

  k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
  if k == 27:
    break
  elif k == ord("c"):
    capture = True
  elif count >= 30:
    break

print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()