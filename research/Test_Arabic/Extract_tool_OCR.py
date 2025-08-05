import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_text_via_ocr(pdf_path):
    """
    Extracts text from a PDF by rendering each page as an image
    and performing OCR. This is effective for problematic PDFs.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    # Iterate over each page
    for page_num in range(len(doc)):
        if page_num==2:
          page = doc.load_page(page_num)

          # 1. Render page to a high-resolution image
          # DPI (dots per inch) is crucial for OCR quality. 300 is a good start.
          pix = page.get_pixmap(dpi=300)
          img_data = pix.tobytes("png")
          image = Image.open(io.BytesIO(img_data))

          # 2. Perform OCR on the image using the Arabic language pack ('ara')
          try:
              page_text = pytesseract.image_to_string(image, lang='ara')
              full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
          except Exception as e:
              print(f"Could not process page {page_num + 1}: {e}")

    return full_text

# --- Usage ---
pdf_file = "Documents/ksa-personal-data-protection-law-series-part-1-ar.pdf"
correctly_extracted_text = extract_text_via_ocr(pdf_file)
with open("Test_Arabic/arabic_page_OCR.txt","a",encoding="utf-8") as f :
   f.write(correctly_extracted_text)