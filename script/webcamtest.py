import os
import pandas as pd
from datetime import datetime
import numpy as np
import cv2
import face_recognition
import dlib
from face_recognition_models import face_recognition_model_location

face_recognition_model = face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

known_face_encodings = []
known_face_names = []
known_face_ids = []

path_to_npy_files = r"C:\Users\class\Desktop\projectfacerecognitionv2\processed\embeddings\all_encoding_person"

for file_name in os.listdir(path_to_npy_files):
    if file_name.endswith('.npy'):
        data = np.load(os.path.join(path_to_npy_files, file_name))
        average_encoding = np.mean(data, axis=0)
        known_face_encodings.append(average_encoding)
        base_name = os.path.splitext(file_name)[0]
        folder_id, name = base_name.split('_', 1)
        known_face_ids.append(folder_id)
        known_face_names.append(name)

known_face_encodings = np.array(known_face_encodings)

for i, enc in enumerate(known_face_encodings):
    print(f"Encoding {i} shape: {enc.shape}")

file_name = "face_data.csv"

face_recognition_model = face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


known_face_encodings = []
known_face_names = []
known_face_ids = []


path_to_npy_files = r"C:\Users\class\Desktop\projectfacerecognitionv2\processed\embeddings\all_encoding_person"

for file_name in os.listdir(path_to_npy_files):
    if file_name.endswith('.npy'):
        data = np.load(os.path.join(path_to_npy_files, file_name))

        average_encoding = np.mean(data, axis=0)
        known_face_encodings.append(average_encoding)

        base_name = os.path.splitext(file_name)[0]
        folder_id, name = base_name.split('_', 1)
        known_face_ids.append(folder_id)
        known_face_names.append(name)


known_face_encodings = np.array(known_face_encodings)


for i, enc in enumerate(known_face_encodings):
    print(f"Encoding {i} shape: {enc.shape}")

file_name = "face_data.csv"  


def detect_faces(frame, known_face_encodings, known_face_names, known_face_ids):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    face_locations = face_recognition.face_locations(rgb_frame)
    print("Detected face locations:", face_locations)

    face_encodings = []

    if face_locations:
        try:
    
            raw_landmarks = face_recognition.api._raw_face_landmarks(rgb_frame, face_locations, model="large")
            print("Raw landmarks:", raw_landmarks)

            if not all(isinstance(landmark, dlib.full_object_detection) for landmark in raw_landmarks):
                raise TypeError("raw_landmarks contains elements that are not of type dlib.full_object_detection")

            for landmark in raw_landmarks:
                descriptor = face_encoder.compute_face_descriptor(rgb_frame, landmark, num_jitters=1)
                face_encodings.append(np.array(descriptor))
            print("Face encodings:", face_encodings)
        except Exception as e:
            print("Error in face encodings:", str(e))

    face_names = []

    for face_encoding in face_encodings:

        face_encoding = np.array(face_encoding)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        folder_id = None

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            folder_id = known_face_ids[best_match_index]

        face_names.append(name)

        if folder_id is not None and name != "Unknown":
            save_face_data(folder_id, name, datetime.now().strftime("%d/%m/%Y %H:%M"))

    return face_locations, face_names


def save_face_data(folder_id, name, timestamp):
    try:
        try:
            existing_data = pd.read_csv(file_name)
        except FileNotFoundError:
            existing_data = pd.DataFrame(columns=["Folder ID", "Name", "Timestamp"])
        if not ((existing_data["Folder ID"] == int(folder_id)) & (existing_data["Name"] == name)).any():
            new_data = pd.DataFrame({"Folder ID": [int(folder_id)], "Name": [name], "Timestamp": [timestamp]})
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            updated_data.to_csv(file_name, index=False)

    except Exception as e:
        print("Error saving face data:", str(e))

def save_face_data(folder_id, name, timestamp):
    try:
        try:
            existing_data = pd.read_csv(file_name)
        except FileNotFoundError:
            existing_data = pd.DataFrame(columns=["Folder ID", "Name", "Timestamp"])

        if not ((existing_data["Folder ID"] == int(folder_id)) & (existing_data["Name"] == name)).any():
            new_data = pd.DataFrame({"Folder ID": [int(folder_id)], "Name": [name], "Timestamp": [timestamp]})
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)

            updated_data.to_csv(file_name, index=False)

    except Exception as e:
        print("Error saving face data:", str(e))

def start_webcam():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        face_locations, face_names = detect_faces(frame, known_face_encodings, known_face_names, known_face_ids)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

start_webcam()
