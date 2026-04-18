import cv2
import numpy as np

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(image_path, output_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Grab the coordinates of the first face found
        (x, y, w, h) = faces[0]
        ih, iw, _ = image.shape
        
        # Add a small margin around the face
        face_crop = image[max(0, y-20):min(ih, y+h+20), max(0, x-20):min(iw, x+w+20)]
        if face_crop.size == 0:
            return None
            
        # Resize for the neural network
        face_resized = cv2.resize(face_crop, output_size)
        # Convert to grayscale for FFT
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        return face_gray
        
    return None