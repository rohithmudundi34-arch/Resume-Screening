from src.skill_extractor import extract_skills


# ==========================================
# SKILL WEIGHTS
# ==========================================

SKILL_WEIGHTS = {

    # Core programming
    "python": 1.5,
    "c": 0.8,
    "c++": 0.8,
    "java": 0.8,

    # Machine Learning / Deep Learning
    "machine learning": 1.5,
    "deep learning": 1.5,
    "scikit-learn": 1.5,
    "tensorflow": 1.5,
    "pytorch": 1.5,
    "xgboost": 1.3,

    # Generative AI
    "llms": 1.5,
    "langchain": 1.5,
    "rag": 1.5,
    "faiss": 1.3,
    "hugging face": 1.3,
    "sentence transformers": 1.3,
    "chromadb": 1.2,

    # NLP
    "nlp": 1.5,

    # Data / ML engineering
    "pandas": 1.2,
    "numpy": 1.2,
    "feature engineering": 1.3,
    "hyperparameter tuning": 1.2,
    "model evaluation": 1.2,

    # Backend
    "flask": 1.0,
    "fastapi": 1.0,
    "rest api": 1.0,

    # Database / Cloud / DevOps
    "sql": 1.0,
    "aws": 0.8,
    "docker": 0.8,
    "git": 0.7,
    "github": 0.7,

    # Visualization
    "matplotlib": 0.8,
    "seaborn": 0.8,
}


def get_skill_weight(skill: str) -> float:
    """
    Return the weight of a skill.
    Default weight = 1.0
    """

    return SKILL_WEIGHTS.get(
        skill.lower(),
        1.0
    )


def calculate_skill_match(
    job_description: str,
    resume_text: str
) -> dict:
    """
    Calculate weighted skill matching.
    """

    job_skills = extract_skills(
        job_description
    )

    resume_skills = extract_skills(
        resume_text
    )

    if not job_skills:

        return {
            "score": 0.0,
            "required_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "matched_weight": 0.0,
            "total_weight": 0.0
        }

    matched_skills = job_skills.intersection(
        resume_skills
    )

    missing_skills = job_skills.difference(
        resume_skills
    )

    # ==========================================
    # WEIGHTED SCORE
    # ==========================================

    total_weight = sum(
        get_skill_weight(skill)
        for skill in job_skills
    )

    matched_weight = sum(
        get_skill_weight(skill)
        for skill in matched_skills
    )

    if total_weight == 0:

        score = 0.0

    else:

        score = (
            matched_weight /
            total_weight
        ) * 100

    return {

        "score": round(score, 2),

        "required_skills": sorted(
            job_skills
        ),

        "matched_skills": sorted(
            matched_skills
        ),

        "missing_skills": sorted(
            missing_skills
        ),

        "matched_weight": round(
            matched_weight,
            2
        ),

        "total_weight": round(
            total_weight,
            2
        )
    }