import zipfile
import os

ZIP_FILE = "archive.zip"
EXTRACT_FOLDER = "dataset"

print(f"Silently extracting 140,000 images to the '{EXTRACT_FOLDER}' folder...")
print("Your laptop will not freeze, but this will take a few minutes. Please wait...")

try:
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    print("\nSUCCESS! All files extracted.") 
except Exception as e:
    print(f"\nError: {e}")