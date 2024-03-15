
## method 1:
from pdfminer.high_level import extract_text

# Path to the Marathi PDF file
pdf_file_path = 'file_path'

# Extract text from the PDF
marathi_text = extract_text(pdf_file_path)

# Print or use the extracted Marathi text as needed
print(marathi_text)



## method 2:
import fitz  # PyMuPDF

# Path to the Marathi PDF file
pdf_file_path = 'path'

# Open the PDF file
pdf_document = fitz.open(pdf_file_path)

# Initialize an empty string to store the extracted text
marathi_text = ''

# Iterate through each page of the PDF and extract text
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    # Use the built-in text extraction with options for better handling
    text_options = {
        'clip': page.rect,  # Clip to the page rectangle
        'text': 'text',     # Extract text in plain format
        'blocks': None,     # Do not separate text into blocks
    }
    marathi_text += page.get_text("text", clip=page.rect)

# Close the PDF document
pdf_document.close()

# Print or use the extracted Marathi text as needed
print(marathi_text)


## method 3:
import fitz  # PyMuPDF

# Path to the Marathi PDF file
pdf_file_path = 'path'

# Open the PDF file (without specifying encoding)
pdf_document = fitz.open(pdf_file_path)

# Initialize an empty string to store the extracted text
marathi_text = ''

# Iterate through each page of the PDF and extract text with UTF-8 decoding
for page_num in range(pdf_document.page_count):
    try:
        page = pdf_document.load_page(page_num)
        text = page.get_text()  # Extract text without specifying encoding
        marathi_text += text.decode("utf-8")  # Decode the extracted text as UTF-8
    except Exception as e:  # Catch potential errors
        print(f"Error extracting text from page {page_num}: {e}")

# Close the PDF document
pdf_document.close()

# Print the extracted Marathi text (if any)
if marathi_text:
    print("Extracted Marathi text:")
    print(marathi_text)
else:
    print("Failed to extract text from the PDF.")

## method 4:
import fitz  # PyMuPDF

# Path to the Marathi PDF file
pdf_file_path = 'path'

# Open the PDF file
pdf_document = fitz.open(pdf_file_path)

# Initialize an empty string to store the extracted text
marathi_text = ''

# Iterate through each page of the PDF and extract text
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    marathi_text += page.get_text()

# Close the PDF document
pdf_document.close()

# Print or use the extracted Marathi text as needed
print(marathi_text)


## method 5:
