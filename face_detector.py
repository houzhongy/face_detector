import cv2

#loading pre-trained data from opencv
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#choose an image to detect face on
img = cv2.imread("face1.jpg")

#Converting the image to grayscale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Drawing rectangles
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

print(face_coordinates)

cv2.imshow("Clever Programmer Face Detector", img)
cv2.waitKey()