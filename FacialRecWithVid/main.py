# Facial Recogition with Video

# Import Libraries
import numpy as np
import cv2
import os
import face_recognition as fr

# Capture video from computer's camera
video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():

    ret, frame = video_capture.read()

    if ret:
        cv2.imshow("Displaying Webcam", frame)

    if cv2.waitKey(25) == 27:
        break

video_capture.release()
cv2.destroyAllWindows()

imagefile = 'FarhanWaterFall.jpg'
image = cv2.imread(imagefile)

cv2.imshow('Farhan',image)
cv2.waitKey(5000)

image_encoded = fr.face_encodings(image)[0]

print('Image Encoded')
print(image_encoded)

print(f'Image Encoded Array Size: {image_encoded.shape}')

known_face = ['Farhan']

while True:
    sucess, videoframe = video_capture.read()
    #Convert VideoFrame to RGB
    videoframe_RGB = cv2.cvtColor(videoframe, cv2.COLOR_BGRA2RGB)

    #Locate faces
    face_locations = fr.face_locations(videoframe_RGB)

    #Encode the faces
    face_encoding = fr.face_encodings(videoframe_RGB, face_locations)

    #Look for matches
    for(top, right, bottom, left),face_encoding in zip(face_locations,face_encoding):
        image_matches = fr.compare_faces(image_encoded, face_encoding)

        match_name = 'Unknown'

        face_distances = fr.face_distance(image_encoded, face_encoding)

        bestmatch_index = np.nanargmin(face_distances)

        if image_matches[bestmatch_index]:
            name = known_face[bestmatch_index]

        cv2.rectangle(videoframe,(left,top),(right,bottom),(0,0,255),2)

        cv2.rectangle(videoframe,(left,bottom -35),(right,bottom,(0,0,255),cv2.FILLED))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(videoframe,name,(left+6,bottom-6),font,1.0,(255,255,255),1)

    cv2.imshow('Webcam Face Recognition',videoframe)

    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



