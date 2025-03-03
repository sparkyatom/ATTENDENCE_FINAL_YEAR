import face_recognition
import cv2
import os
import numpy as np
import pickle
import random
import datetime
import time
from tqdm import tqdm
import ast
from datetime import datetime
class FaceRecognitionSystem:
    def __init__(self, data_dir=None, model_path=None):
        self.data_dir = data_dir
        self.model_path = model_path
        self.known_face_encodings = []
        self.known_face_names = []

    def load_and_encode_faces(self):
        print("Loading and encoding faces...")

        for person in tqdm(os.listdir(self.data_dir)):
            person_dir = os.path.join(self.data_dir, person)
            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(person_dir, img_name)

                try:
                    image = face_recognition.load_image_file(img_path)
                    face_encodings = face_recognition.face_encodings(image)

                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(person)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    def save_model(self):
        if not self.model_path:
            raise ValueError("Model path is not specified.")

        model_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if not self.model_path:
            raise ValueError("Model path is not specified.")

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.known_face_encodings = model_data['encodings']
        self.known_face_names = model_data['names']

        print(f"Model loaded from {self.model_path}")

    def predict_face(self, image_path, tolerance=0.6):
        try:
            unknown_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(unknown_image)

            if not face_encodings:
                return "No face detected"

            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] <= tolerance:
                    return self.known_face_names[best_match_index]

            return "Unknown"

        except Exception as e:
            print(f"Error processing image at {image_path}: {e}")
            return None

    def run_complete_pipeline(self, new_image_path=None):
        if self.data_dir:
            self.load_and_encode_faces()
            if self.model_path:
                self.save_model()

        if new_image_path:
            prediction = self.predict_face(new_image_path)
            print(f"Predicted person: {prediction}")
            return prediction

        return None

def extract_random_frames(master_folder, output_folder,list, frame_count=100):

    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(master_folder):
        video_path = os.path.join(master_folder, video_file)

        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_name = os.path.splitext(video_file)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < frame_count:
            print(f"Skipping {video_file}, not enough frames.")
            cap.release()
            continue

        random_frames = random.sample(range(total_frames), frame_count)
        random_frames.sort()

        frame_idx = 0
        extracted = 0

        while extracted < frame_count and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in random_frames:
                frame_path = os.path.join(video_output_folder, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted += 1

            frame_idx += 1

        cap.release()
        
        print(f"Extracted {extracted} frames from {video_file}.")
        list.append(video_file)

def predict_and_display_images(model_path, test_image_folder, list,tolerance=0.6):
    """
    Predict the identity of a person in images from a folder and display the results.

    :param model_path: Path to the trained model file.
    :param test_image_folder: Path to the folder containing test images.
    :param tolerance: Distance threshold for face matching. Lower is stricter.
    """
    # Load the trained model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    known_face_encodings = model_data['encodings']
    known_face_names = model_data['names']

    # Iterate through all images in the test folder
    for img_name in os.listdir(test_image_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        img_path = os.path.join(test_image_folder, img_name)
        print(f"Processing image: {img_name}")

        try:
            # Load the image
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                print(f"No face detected in {img_name}")
                continue

            # Convert the image to BGR format for OpenCV display
            image_display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Loop through each face found in the image
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare the face with known faces
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] <= tolerance:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"

                # Draw a rectangle around the face
                cv2.rectangle(image_display, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw the predicted name below the face
                #cv2.putText(image_display, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(image_display, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                print(f"Predicted person in {img_name}: {name}")
                list.append(name)
                

            # Display the image with predictions
            cv2.imshow("Predicted Image", image_display)
            cv2.waitKey(0)  # Wait for a key press to move to the next image
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
def crop_faces(image_path, output_folder="cropped_faces", min_face_size=(30, 30)):
    """
    Detect and crop faces from an image, saving each face as a separate file.
    
    Args:
        image_path (str): Path to the input image
        output_folder (str): Folder to save cropped faces
        min_face_size (tuple): Minimum size of face to detect (width, height)
    
    Returns:
        int: Number of faces detected and saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return 0
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_face_size
    )
    
    print(f"Found {len(faces)} faces!")
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crop and save each face
    for i, (x, y, w, h) in enumerate(faces):
        # Add some margin around the face
        margin = int(min(w, h) * 0.2)
        x_margin = max(0, x - margin)
        y_margin = max(0, y - margin)
        w_margin = min(image.shape[1] - x_margin, w + 2 * margin)
        h_margin = min(image.shape[0] - y_margin, h + 2 * margin)
        
        # Crop the face
        face = image[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
        
        # Save the cropped face
        face_filename = os.path.join(output_folder, f"face_{timestamp}_{i+1}.jpg")
        cv2.imwrite(face_filename, face)
        print(f"Saved face {i+1} to {face_filename}")
    
    return len(faces)

def cropping(image_path):
    
    # Validate image path
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return
    
    # Process the image
    num_faces = crop_faces(image_path)
    
    if num_faces > 0:
        print(f"Successfully saved {num_faces} faces to the 'cropped_faces' folder.")
    else:
        print("No faces were detected in the image.")
def remove_duplicates(lst):
    if not isinstance(lst, list):
        raise TypeError("Input must be a list")

    seen = set()
    unique_list = []

    for item in lst:
        if isinstance(item, str) and item not in seen:
            seen.add(item)
            unique_list.append(item)

    return unique_list
if __name__ == "__main__":
    person_names=[]
    master_folder = "video_data"
    output_folder = "phase_final_op"

    if not os.path.exists(output_folder) or not os.listdir(output_folder):
        extract_random_frames(master_folder, output_folder,person_names)

    model_path = "face_recognition_model.pkl"
    data_dir = output_folder

    start_time = time.time()

    face_recognition_system = FaceRecognitionSystem(data_dir=data_dir, model_path=model_path)

    if os.path.exists(model_path):
        face_recognition_system.load_model()
    else:
        print("Training model...")
        face_recognition_system.run_complete_pipeline()

    test_image_path = "test_image.png"
    face_recognition_system.predict_face(test_image_path)

    end_time = time.time()

    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    ######
    print("------CROPPING STARTED-----")
    cropping(r"C:\Users\sande\Desktop\projectfinal_github\ATTENDENCE_FINAL_YEAR\final_year_project\tofinalize\group_image.jpg")#grouppics input 
    # Example usage
    model_path = r"C:\Users\sande\Desktop\projectfinal_github\ATTENDENCE_FINAL_YEAR\final_year_project\tofinalize\face_recognition_model.pkl"
    test_image_folder = "cropped_faces"#input's crop face
    l=[]
    
    predict_and_display_images(model_path, test_image_folder,l)
    #l= predicted persons list
    predicted_person_list=remove_duplicates(l)
    with open('person_details.txt', 'w') as f:
      f.write(f"LIST OF PERSONS:\n {person_names}\n")
      f.write(f"NUMBER OF PERSONS:\n {len(person_names)}\n")
    with open('model_details.txt', 'w') as f:
      f.write(f"LIST OF PERSONS:\n {person_names}\n")
      f.write(f"TOTAL PERSONS:\n {len(person_names)}\n")
      f.write(f"TOTAL TIME REQUIRED TO TRAIN IT {end_time - start_time:.2f} SECONDS")
    with open('PREDICTED_PERSONS.txt', 'w') as f:
      f.write(f"LIST OF PERSONS WHICH ARE IN THE GROUP PHOTO:\n {predicted_person_list}\n")
      f.write(f"NUMBER OF PERSONS:\n {len(predicted_person_list)}\n")

