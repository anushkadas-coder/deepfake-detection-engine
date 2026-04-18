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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def perform_inference(image_pil):
    # STEP 1: SOLVING NOISE - Convert to CV2 to de-noise
    open_cv_image = np.array(image_pil) 
    # Applying a Gaussian Blur to remove Moire patterns and phone sensor noise
    denoised = cv2.GaussianBlur(open_cv_image, (3, 3), 0)
    image_pil = Image.fromarray(denoised)

    img_tensor = transform(image_pil).unsqueeze(0).to(device)
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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                score = perform_inference(pil_img)
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
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            fake_prob = perform_inference(image)

        # STEP 2: SOLVING ACCURACY - Higher threshold for FAKE
        # Only flag as FAKE if the confidence is definitively high (>= 75%)
        if fake_prob >= 75.0:
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
    uvicorn.run(app, host="127.0.0.1", port=8080)