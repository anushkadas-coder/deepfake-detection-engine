import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

# ---------------------------------------------------------
# 1. THE ARCHITECTURE (Must match training exactly)
# ---------------------------------------------------------
class ResNetDeepfakeDetector(nn.Module):
    def __init__(self):
        super(ResNetDeepfakeDetector, self).__init__()
        # Load the ResNet18 architecture without default weights
        self.resnet = models.resnet18(weights=None) 
        
        # Modify the final layer to match our custom 1-output design
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.resnet(x)

# ---------------------------------------------------------
# 2. THE MATH (Must match training exactly)
# ---------------------------------------------------------
def extract_face(image_path, output_size=(224, 224)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    if image is None: return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        ih, iw, _ = image.shape
        face_crop = image[max(0, y-20):min(ih, y+h+20), max(0, x-20):min(iw, x+w+20)]
        if face_crop.size == 0: return None
        face_resized = cv2.resize(face_crop, output_size)
        return cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    return None

def generate_fft_spectrum(gray_image):
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    img_3c = np.stack((magnitude_normalized,)*3, axis=-1)
    return np.transpose(img_3c, (2, 0, 1)) / 255.0

# ---------------------------------------------------------
# 3. THE PREDICTION ENGINE
# ---------------------------------------------------------
def predict_image(image_path, model_path="best_resnet_fft.pth"):
    device = torch.device("cpu")
    
    # 1. Load your saved "brain"
    model = ResNetDeepfakeDetector().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Make sure it is in the same folder!")
        return

    # 2. Process the image
    print(f"Scanning: {image_path}...")
    face_gray = extract_face(image_path)
    
    if face_gray is None:
        print("Result: No human face detected in this image!")
        return

    # 3. Apply FFT and convert to tensor
    fft_tensor = generate_fft_spectrum(face_gray)
    input_tensor = torch.tensor(fft_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    # 4. Make the prediction
    with torch.no_grad():
        output = model(input_tensor)
        score = output.item() 

    # 5. Interpret the math (0.0 is Real, 1.0 is Fake)
    if score > 0.5:
        confidence = score * 100
        print(f"\n🚨 RESULT: FAKE (AI Generated)")
        print(f"Confidence: {confidence:.2f}%")
    else:
        confidence = (1.0 - score) * 100
        print(f"\n✅ RESULT: REAL (Human)")
        print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    # 🚨 CHANGE THIS PATH to the image you want to test!
    TEST_IMAGE = "dataset/fake/fake_face (1).webp" 
    
    predict_image(TEST_IMAGE)