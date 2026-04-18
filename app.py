import cv2
import torch
import torch.nn as nn
import numpy as np
import tempfile
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path="best_resnet_fft.pth"):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256), 
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )
        
    try:
        state_dict = torch.load(weights_path, map_location=device)
        new_state_dict = {k.replace("resnet.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"ERROR: {e}")
        return None

model = load_model()

# Notice we use ToPILImage here because cv2 outputs numpy arrays
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def perform_inference(image_bytes):
    # 1. Load image in grayscale directly from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # 2. MICRO-BLUR (3x3): Removes sensor noise but KEEPS deepfake artifacts
    img_smoothed = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 3. THE MISSING MATH: 2D Fast Fourier Transform (FFT)
    f = np.fft.fft2(img_smoothed)
    fshift = np.fft.fftshift(f)
    
    # 4. Enhance the Magnitude Spectrum so the ResNet can "see" the anomalies
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # 5. Normalize for ResNet-18 (0-255) and convert to RGB
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_spectrum = np.uint8(magnitude_spectrum)
    img_rgb = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)
    
    # 6. Pass the Frequency Spectrum to the Model
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item() * 100
        
    return probability

def process_video(video_bytes):
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_scores = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % fps == 0:
                # Convert frame back to bytes for our new FFT inference pipeline
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    score = perform_inference(buffer.tobytes())
                    frame_scores.append(score)
            count += 1
            if len(frame_scores) >= 15: break
        cap.release()
    return sum(frame_scores) / len(frame_scores) if frame_scores else 0

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Engine offline.")
    
    contents = await file.read()
    filename = file.filename.lower()
    
    try:
        if filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
            fake_prob = process_video(contents)
        else:
            fake_prob = perform_inference(contents)
            
        # STEP 2: AGGRESSIVE THRESHOLD
        # We lowered this to 40% to catch modern Diffusion model fakes
        if fake_prob >= 40.0:
            prediction = "FAKE"
            confidence = fake_prob
        else:
            prediction = "REAL"
            confidence = 100 - fake_prob
                
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Hugging Face requires port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)