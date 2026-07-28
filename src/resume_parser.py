import pdfplumber
from pathlib import Path

class ResumeParser:
    """
    Handles extraction of text from text-based PDF resumes.
    """
    def __init__(self,pdf_path):
        self.pdf_path=Path(pdf_path)
    
    def extract_text(self):
            """
            Extracts all text from the PDF and returns it as a string."""
            
            with pdfplumber.open(self.pdf_path) as pdf:
                 text=[]
                
                 for page in pdf.pages:
                      page_text = page.extract_text()
                      if page_text:
                           text.append(page_text)
                           
            return "\n".join(text)