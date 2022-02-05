import cv2

#loading pre-trained data from opencv
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

#iterate forvever for all the frames
while True:
    #Reading the current frame
    successful_frame_read, frame = webcam.read()


    #Converting the image to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Drawing rectangles
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    print(face_coordinates)

    cv2.imshow("Clever Programmer Face Detector", frame)
    key = cv2.waitKey(1)

    #break if "esc" is pressed
    if key == 27:
        break