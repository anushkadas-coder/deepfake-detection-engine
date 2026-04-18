import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.models as models

# ==========================================
# 1. PREPROCESSING & FFT MODULES
# ==========================================
def extract_face(image_path, output_size=(224, 224)):
    # Using the fast OpenCV cascade
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
    # Convert to 3 channels so it fits ResNet-18
    img_3c = np.stack((magnitude_normalized,)*3, axis=-1)
    # PyTorch expects shape [Channels, Height, Width]
    return np.transpose(img_3c, (2, 0, 1)) / 255.0

# ==========================================
# 2. DATASET MODULE
# ==========================================
class DeepfakeFFTDataset(Dataset):
    def __init__(self, image_paths, labels, is_train=False):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        face_gray = extract_face(img_path)
        if face_gray is None: return None 

        if self.is_train:
            # Data Augmentation to prevent overfitting
            if random.random() > 0.5: face_gray = cv2.flip(face_gray, 1)
            if random.random() > 0.7: face_gray = cv2.GaussianBlur(face_gray, (3, 3), 0)

        fft_tensor = generate_fft_spectrum(face_gray)
        return torch.tensor(fft_tensor, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

# ==========================================
# 3. MODEL ARCHITECTURE (ResNet-18)
# ==========================================
class ResNetDeepfakeDetector(nn.Module):
    def __init__(self):
        super(ResNetDeepfakeDetector, self).__init__()
        # Load the massive, pre-built ResNet18 architecture
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the final layer to output 1 prediction (Fake/Real)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevents overfitting
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

# ==========================================
# 4. CPU-OPTIMIZED TRAINING ENGINE
# ==========================================
def train_model():
    # Correctly pointed to your new dataset folders
    REAL_DIR = os.path.join("dataset", "Dataset", "Train", "Real") 
    FAKE_DIR = os.path.join("dataset", "Dataset", "Train", "Fake")

    # Limiting to 500 images per category so your laptop CPU doesn't melt. 
    # If your computer handles this easily, you can increase this number to 1000 or 5000 later!
    real_paths = glob.glob(os.path.join(REAL_DIR, "*.*"))[:500]
    fake_paths = glob.glob(os.path.join(FAKE_DIR, "*.*"))[:500]

    all_paths = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

    if len(all_paths) == 0:
        print("Error: No images found. Check your REAL_DIR and FAKE_DIR paths!")
        return

    print(f"Loaded {len(real_paths)} REAL and {len(fake_paths)} FAKE images.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42)

    # Batch size lowered to 16 to save laptop RAM
    train_loader = DataLoader(DeepfakeFFTDataset(train_paths, train_labels, is_train=True), batch_size=16, shuffle=True, collate_fn=safe_collate)
    val_loader = DataLoader(DeepfakeFFTDataset(val_paths, val_labels, is_train=False), batch_size=16, shuffle=False, collate_fn=safe_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ignition. Training locally on: {device} with ResNet-18")

    model = ResNetDeepfakeDetector().to(device)
    criterion = nn.BCELoss()
    
    # Slightly smaller learning rate because ResNet is highly sensitive
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    EPOCHS = 10
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        for inputs, labels in loop:
            if inputs.numel() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += ((outputs > 0.5).float() == labels).sum().item()
            train_total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs.numel() == 0: continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_correct += ((outputs > 0.5).float() == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        print(f"\nResult Epoch {epoch+1} -> Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if avg_val_loss < best_loss and val_total > 0:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_resnet_fft.pth")
            print(">>> Model improved and saved! <<<")

    print("\nDONE! Your professional-grade model is saved as 'best_resnet_fft.pth'.")

if __name__ == "__main__":
    train_model()