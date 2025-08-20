import re
from pdfminer.high_level import extract_text

class LanguageDetector:
    """Language detector for Arabic and English"""
    
    def __init__(self):
        # Arabic Unicode ranges
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        # English pattern (basic Latin characters)
        self.english_pattern = re.compile(r'[a-zA-Z]')
    
    def detect_language(self, question: str = None, pdf_path: str = None, pages_to_check: int = 3) -> str:
        """Detect language from text or PDF"""
        arabic_count, english_count = 0, 0
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        english_pattern = re.compile(r'[A-Za-z]')
        
        # Extract first few pages of text
        if pdf_path:
            text = extract_text(pdf_path, maxpages=pages_to_check)
            if not text.strip():
                return "no_text"
        elif question:
            text = question
        else:
            return "unknown"
            
        # Count characters
        arabic_count = len(arabic_pattern.findall(text))
        english_count = len(english_pattern.findall(text))
        
        if arabic_count > 0:
            return "ar"
        elif english_count > arabic_count:
            return "en"
        else:
            return "unknown"