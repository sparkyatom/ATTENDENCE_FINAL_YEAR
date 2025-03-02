from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import os
from deepface import DeepFace

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Path to student images
STUDENT_DB_PATH = "students/"

def recognize_face(image_data):
    """Process the face and compare it with stored student images"""
    try:
        # Convert Base64 image data to OpenCV format
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Use DeepFace to compare with student images
        result = DeepFace.find(img_path=img, db_path=STUDENT_DB_PATH, enforce_detection=False)

        if len(result) > 0:
            student_name = os.path.basename(result[0]['identity'][0]).split(".")[0]  # Extract student name
            return f"{student_name} marked present!"
        else:
            return "Face not recognized. Try again!"
    
    except Exception as e:
        return f"Error: {str(e)}"

@socketio.on("send_face")
def handle_face(data):
    """Handle face recognition request from frontend"""
    recognition_result = recognize_face(data["image"])
    emit("attendance_status", {"status": recognition_result})  # Send result back

if __name__ == "__main__":
    if not os.path.exists(STUDENT_DB_PATH):
        os.makedirs(STUDENT_DB_PATH)  # Create student folder if not exists
    socketio.run(app, debug=True)
