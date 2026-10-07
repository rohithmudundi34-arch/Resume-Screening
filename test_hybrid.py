from pathlib import Path

from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text

from src.keyword_matcher import calculate_bm25_scores
from src.semantic_matcher import calculate_semantic_scores
from src.skill_matcher import calculate_skill_match
from src.profile_matcher import calculate_profile_match
from src.hybrid_scorer import calculate_hybrid_score


# ============================================================
# LOAD JOB DESCRIPTION
# ============================================================

job_description = Path(
    "data/job_description.txt"
).read_text(
    encoding="utf-8"
)

job_description = clean_text(
    job_description
)


# ============================================================
# LOAD RESUMES
# ============================================================

resume_files = sorted(
    Path("data").glob("*.pdf")
)

resumes = []
resume_names = []


for resume_file in resume_files:

    print(
        f"Reading: {resume_file.name}"
    )

    text = extract_text_from_pdf(
        str(resume_file)
    )

    text = clean_text(
        text
    )

    resumes.append(text)

    resume_names.append(
        resume_file.name
    )


print(
    f"\nFound {len(resumes)} resumes"
)


# ============================================================
# BM25
# ============================================================

bm25_scores = calculate_bm25_scores(
    job_description,
    resumes
)


# ============================================================
# SEMANTIC
# ============================================================

semantic_scores = calculate_semantic_scores(
    job_description,
    resumes
)


# ============================================================
# SKILL + PROFILE MATCH
# ============================================================

skill_scores = []
profile_results = []


for resume in resumes:

    # --------------------------------------------------------
    # SKILL MATCH
    # --------------------------------------------------------

    skill_result = calculate_skill_match(
        job_description,
        resume
    )

    skill_scores.append(
        skill_result["score"]
    )

    # --------------------------------------------------------
    # PROFILE MATCH
    # --------------------------------------------------------

    profile_result = calculate_profile_match(
        job_description,
        resume
    )

    profile_results.append(
        profile_result
    )


# ============================================================
# HYBRID SCORE
# ============================================================

results = []


for i, name in enumerate(resume_names):

    profile_score = profile_results[i][
        "score"
    ]

    final_score = calculate_hybrid_score(
        bm25_scores[i],
        semantic_scores[i],
        skill_scores[i],
        profile_score
    )

    results.append({

        "name": name,

        "bm25": bm25_scores[i],

        "semantic": semantic_scores[i],

        "skills": skill_scores[i],

        "profile": profile_score,

        "education": profile_results[i][
            "education_score"
        ],

        "role": profile_results[i][
            "role_score"
        ],

        "experience": profile_results[i][
            "experience_score"
        ],

        "required_years": profile_results[i][
            "required_years"
        ],

        "candidate_years": profile_results[i][
            "candidate_years"
        ],

        "final": final_score
    })


# ============================================================
# SORT BY FINAL SCORE
# ============================================================

results.sort(
    key=lambda x: x["final"],
    reverse=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n========== HYBRID RECOMMENDATION ==========\n"
)


for rank, result in enumerate(
    results,
    start=1
):

    print(
        f"{rank}. {result['name']}"
    )

    print(
        f"   BM25:          {result['bm25']}%"
    )

    print(
        f"   Semantic:      {result['semantic']}%"
    )

    print(
        f"   Skills:        {result['skills']}%"
    )

    print(
        f"   Profile:       {result['profile']}%"
    )

    print(
        f"   Education:     {result['education']}%"
    )

    print(
        f"   Role:          {result['role']}%"
    )

    print(
        f"   Experience:    {result['experience']}%"
    )

    print(
        f"   Required Exp:  "
        f"{result['required_years']} years"
    )

    print(
        f"   Candidate Exp: "
        f"{result['candidate_years']} years"
    )

    print(
        f"   FINAL:         {result['final']}%"
    )

    print()