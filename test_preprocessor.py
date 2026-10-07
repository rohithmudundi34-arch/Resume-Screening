from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text


pdf_path = "data/YOGESHEWAR .pdf"

# Extract PDF text
text = extract_text_from_pdf(pdf_path)

# Clean text
cleaned_text = clean_text(text)

print("\n========== CLEANED RESUME ==========\n")
print(cleaned_text[:3000])