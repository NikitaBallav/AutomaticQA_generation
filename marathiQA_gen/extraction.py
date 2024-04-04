import os
import pytesseract
from PIL import Image

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C://Program Files//Tesseract-OCR//tesseract.exe'

# Path to the folder containing PNG files
folder_path = 'C://chatbot nic//1966 marathidata folder//png'

# Output directory for saving text files
output_dir = 'C://output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all files in the folder
print("Listing files in the folder...")
files = os.listdir(folder_path)

# Filter only PNG files
print("Filtering PNG files...")
png_files = [file for file in files if file.endswith('.png')]

# Sort PNG files alphabetically or numerically
print("Sorting PNG files...")
sorted_png_files = sorted(png_files)  # Alphabetical sorting

# Loop through each PNG file
print("Processing PNG files...")
for png_file in sorted_png_files:
    # Extract text using Tesseract OCR
    print(f"Extracting text from {png_file}...")
    png_path = os.path.join(folder_path, png_file)
    text = pytesseract.image_to_string(Image.open(png_path), lang='mar+eng', config='--psm 6')
    
    # Save extracted text as a separate text file
    output_text_file = os.path.join(output_dir, f"{os.path.splitext(png_file)[0]}.txt")
    with open(output_text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Extracted text saved to {output_text_file}")

print("All text extraction completed.")
