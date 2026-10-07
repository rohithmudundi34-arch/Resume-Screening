from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text
from src.profile_matcher import (
    extract_date_ranges,
    extract_resume_experience
)


# ============================================================
# RESUME
# ============================================================

resume_path = "data/YOGESHEWAR .pdf"


# ============================================================
# EXTRACT PDF
# ============================================================

raw_text = extract_text_from_pdf(
    resume_path
)


# ============================================================
# PRINT RAW TEXT
# ============================================================

print("\n========== RAW PDF TEXT ==========\n")

print(raw_text)


# ============================================================
# CLEAN TEXT
# ============================================================

cleaned_text = clean_text(
    raw_text
)


# ============================================================
# PRINT CLEANED TEXT
# ============================================================

print("\n========== CLEANED TEXT ==========\n")

print(cleaned_text)


# ============================================================
# DATE RANGES
# ============================================================

ranges = extract_date_ranges(
    cleaned_text
)

print("\n========== DATE RANGES ==========\n")

if not ranges:

    print("NO DATE RANGES FOUND")

else:

    for start, end in ranges:

        print(
            f"Start: {start} | End: {end}"
        )


# ============================================================
# EXPERIENCE
# ============================================================

experience = extract_resume_experience(
    cleaned_text
)

print("\n========== EXPERIENCE ==========\n")

print(
    f"Candidate Experience: "
    f"{experience} years"
)