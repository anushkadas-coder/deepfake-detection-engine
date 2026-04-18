import cv2
import torch
import numpy as np
from fft_processor import generate_fft_spectrum

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_video(video_path, model, device, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    model.eval() 
    with torch.no_grad():
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every Nth frame
            if frame_count % frame_skip == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    ih, iw, _ = frame.shape
                    
                    face_crop = frame[max(0, y-20):min(ih, y+h+20), max(0, x-20):min(iw, x+w+20)]
                    if face_crop.size > 0:
                        face_resized = cv2.resize(face_crop, (224, 224))
                        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                        
                        fft_data = generate_fft_spectrum(face_gray)
                        tensor_input = torch.tensor(fft_data, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        pred = model(tensor_input).item()
                        predictions.append(pred)
            
            frame_count += 1

    cap.release()
    if not predictions:
        return "No faces detected."
        
    avg_score = np.mean(predictions)
    is_fake = avg_score > 0.5
    
    return {
        "is_fake": is_fake,
        "confidence_score": round(avg_score, 4),
        "frames_analyzed": len(predictions)
    }