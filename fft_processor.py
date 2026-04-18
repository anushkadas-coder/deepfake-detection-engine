import numpy as np
import cv2

def generate_fft_spectrum(gray_face):
    # 1. Compute 2D FFT
    f_transform = np.fft.fft2(gray_face)
    
    # 2. Shift zero-frequency component to the center
    f_shift = np.fft.fftshift(f_transform)
    
    # 3. Calculate Magnitude Spectrum (Log scale to compress the massive dynamic range)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
    
    # 4. Normalize the spectrum to be between 0 and 255 so the CNN can read it
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 5. Reshape to add a channel dimension for PyTorch (1, 224, 224)
    tensor_input = np.expand_dims(magnitude_spectrum, axis=0) 
    
    return tensor_input