from src.pdf_parser import extract_text_from_pdf


pdf_path = "data/YOGESHEWAR .pdf"

text = extract_text_from_pdf(pdf_path)

print(text[:3000])