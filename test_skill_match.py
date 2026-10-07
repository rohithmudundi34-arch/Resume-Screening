from pathlib import Path

from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text
from src.skill_matcher import calculate_skill_match


# ==========================================
# JOB DESCRIPTION
# ==========================================

job_description = Path(
    "data/job_description.txt"
).read_text(
    encoding="utf-8"
)

job_description = clean_text(
    job_description
)


# ==========================================
# RESUME
# ==========================================

resume_path = "data/YOGESHEWAR .pdf"

resume_text = extract_text_from_pdf(
    resume_path
)

resume_text = clean_text(
    resume_text
)


# ==========================================
# SKILL MATCH
# ==========================================

result = calculate_skill_match(
    job_description,
    resume_text
)


print("\n========== SKILL MATCH ==========\n")

print(
    f"Skill Match: {result['score']}%"
)

print("\nRequired Skills:")

for skill in result["required_skills"]:
    print(f"  • {skill}")


print("\nMatched Skills:")

for skill in result["matched_skills"]:
    print(f"  ✓ {skill}")


print("\nMissing Skills:")

for skill in result["missing_skills"]:
    print(f"  ✗ {skill}")