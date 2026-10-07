from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text
from src.skill_extractor import extract_skills


pdf_path = "data/YOGESHEWAR .pdf"

# Extract
text = extract_text_from_pdf(pdf_path)

# Clean
cleaned_text = clean_text(text)

# Extract skills
skills = extract_skills(cleaned_text)

print("\n========== EXTRACTED SKILLS ==========\n")

for skill in sorted(skills):
    print(f"✓ {skill}")