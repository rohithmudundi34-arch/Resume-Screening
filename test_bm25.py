from pathlib import Path

from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text
from src.keyword_matcher import calculate_bm25_scores


# ==========================================
# JOB DESCRIPTION
# ==========================================

job_path = Path("data/job_description.txt")

job_description = job_path.read_text(
    encoding="utf-8"
)

job_description = clean_text(job_description)


# ==========================================
# LOAD ALL RESUMES
# ==========================================

resume_folder = Path("data")

resume_files = sorted(
    resume_folder.glob("*.pdf")
)

print(f"\nFound {len(resume_files)} resumes")


resumes = []
resume_names = []


for resume_file in resume_files:

    print(f"Reading: {resume_file.name}")

    text = extract_text_from_pdf(
        str(resume_file)
    )

    text = clean_text(text)

    resumes.append(text)
    resume_names.append(resume_file.name)


# ==========================================
# BM25 MATCHING
# ==========================================

scores = calculate_bm25_scores(
    job_description,
    resumes
)


# ==========================================
# RANKING
# ==========================================

results = list(
    zip(resume_names, scores)
)

results.sort(
    key=lambda x: x[1],
    reverse=True
)


print("\n========== BM25 RANKING ==========\n")


for rank, (name, score) in enumerate(
    results,
    start=1
):

    print(
        f"{rank}. {name} → {score}%"
    )