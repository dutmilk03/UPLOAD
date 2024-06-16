import os
import face_recognition
import numpy as np
import shutil
from PIL import Image

def process_faces(data_dir, processed_faces_dir, processed_embeddings_dir):
    all_encoding_person_dir = os.path.join(processed_embeddings_dir, "all_encoding_person")
    os.makedirs(all_encoding_person_dir, exist_ok=True)

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        person_faces_dir = os.path.join(processed_faces_dir, f"faces_{person_name}")
        person_embeddings_dir = os.path.join(processed_embeddings_dir, f"encodings_{person_name}")

        os.makedirs(person_faces_dir, exist_ok=True)
        os.makedirs(person_embeddings_dir, exist_ok=True)

        all_face_encodings = []
        face_count = 0

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            print(f"Processing file: {image_path}")

            try:
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, face_locations)
                print(f"Found {len(face_locations)} face(s) in {image_path}")

                for face_location, face_encoding in zip(face_locations, encodings):
                    if face_count >= 6:
                        break
                            
                    top, right, bottom, left = face_location
                    face_image = image[top:bottom, left:right]
                    face_image_pil = Image.fromarray(face_image)
                    face_image_filename = f"{person_name}_face{face_count + 1}.jpg"
                    face_image_path = os.path.join(person_faces_dir, face_image_filename)
                    face_image_pil.save(face_image_path)

                    all_face_encodings.append(face_encoding)
                    face_count += 1

            except Exception as e:
                print(f"Error processing file {image_path}: {e}")

        if all_face_encodings:
            all_face_encodings_path = os.path.join(person_embeddings_dir, f"{person_name}.npy")
            np.save(all_face_encodings_path, all_face_encodings)
            print(f"Saved all encodings for {person_name} to {all_face_encodings_path}")
            
            shutil.move(all_face_encodings_path, os.path.join(all_encoding_person_dir, f"{person_name}.npy"))
            print(f"Moved {person_name}.npy to {all_encoding_person_dir}")

data_dir = r"C:\Users\class\Desktop\projectfacerecognitionv2\dataset"
processed_faces_dir = r"C:\Users\class\Desktop\projectfacerecognitionv2\processed\faces"
processed_embeddings_dir = r"C:\Users\class\Desktop\projectfacerecognitionv2\processed\embeddings"

os.makedirs(processed_faces_dir, exist_ok=True)
os.makedirs(processed_embeddings_dir, exist_ok=True)

process_faces(data_dir, processed_faces_dir, processed_embeddings_dir)
