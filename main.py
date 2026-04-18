import os
import torch
import cv2
import matplotlib.pyplot as plt

# Import from your custom modules
from preprocess import extract_face
from fft_processor import generate_fft_spectrum
from model import get_fft_cnn
from video_detector import predict_video

def process_single_image(image_path, model, device):
    print(f"Processing image: {image_path}")
    
    # 1. Extract Face
    face_gray = extract_face(image_path)
    if face_gray is None:
        print("Error: No face detected in the image.")
        return
        
    # 2. Generate FFT
    fft_tensor = generate_fft_spectrum(face_gray)
    
    # 3. Prepare for Model
    tensor_input = torch.tensor(fft_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 4. Predict
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_input).item()
        
    is_fake = prediction > 0.5
    print(f"Result: {'FAKE' if is_fake else 'REAL'} (Confidence: {prediction:.4f})")
    
    # Optional: Visualize the FFT
    plt.imshow(fft_tensor[0], cmap='gray')
    plt.title(f"FFT Spectrum - {'FAKE' if is_fake else 'REAL'}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    print("Loading CNN Model...")
    model = get_fft_cnn().to(device)
    
    # NOTE: If you have trained weights, load them here like this:
    # model.load_state_dict(torch.load("my_model_weights.pth"))
    
    # Choose your test mode: "image" or "video"
    TEST_MODE = "image" 
    
    if TEST_MODE == "image":
        # Using the exact filename from your VS Code sidebar
        test_img = "test_image.webp" 
        
        if os.path.exists(test_img):
            process_single_image(test_img, model, device)
        else:
            print(f"Error: Could not find '{test_img}' in your folder.")
            
    elif TEST_MODE == "video":
        test_vid = "test_video.mp4"
        if os.path.exists(test_vid):
            print(f"Analyzing video: {test_vid}")
            results = predict_video(test_vid, model, device)
            print(results)
        else:
            print(f"Error: Could not find video '{test_vid}' in your folder.")