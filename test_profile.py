from pathlib import Path

from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text
from src.profile_matcher import calculate_profile_match


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
# PROFILE MATCH
# ==========================================

result = calculate_profile_match(
    job_description,
    resume_text
)


# ==========================================
# OUTPUT
# ==========================================

print(
    "\n========== PROFILE MATCH ==========\n"
)


print(
    f"Overall Profile: "
    f"{result['score']}%"
)


print(
    f"Education Match: "
    f"{result['education_score']}%"
)


print(
    f"Role Match: "
    f"{result['role_score']}%"
)


print(
    f"Experience Match: "
    f"{result['experience_score']}%"
)


# ==========================================
# EXPERIENCE DETAILS
# ==========================================

print(
    f"\nRequired Experience: "
    f"{result['required_years']} years"
)


print(
    f"Candidate Experience: "
    f"{result['candidate_years']} years"
)


# ==========================================
# EDUCATION
# ==========================================

print("\nJob Education:")

for item in result["job_education"]:

    print(
        f"  • {item}"
    )


print("\nResume Education:")

for item in result["resume_education"]:

    print(
        f"  ✓ {item}"
    )


# ==========================================
# ROLES
# ==========================================

print("\nJob Roles:")

for item in result["job_roles"]:

    print(
        f"  • {item}"
    )


print("\nResume Roles:")

for item in result["resume_roles"]:

    print(
        f"  ✓ {item}"
    )