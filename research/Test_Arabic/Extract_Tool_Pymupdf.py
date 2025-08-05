import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        if page_num==2:
          page = doc.load_page(page_num)
          # The 'text' method in PyMuPDF is generally good at
          # handling RTL text reconstruction.
          full_text += page.get_text() + "\n"
    return full_text

# --- Usage ---
pdf_file = "Documents/ksa-personal-data-protection-law-series-part-1-ar.pdf"
correctly_extracted_text = extract_text_from_pdf(pdf_file)
with open("Test_Arabic/arabic_page.txt","a",encoding="utf-8") as f :
   f.write(correctly_extracted_text)