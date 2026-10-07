import re
import numpy as np
from rank_bm25 import BM25Okapi
def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    """

    return re.findall(
        r"[a-zA-Z0-9+#.-]+",
        text.lower()
    )


def calculate_bm25_scores(
    job_description: str,
    resumes: list[str]
) -> list[float]:
    """
    Calculate BM25 scores for multiple resumes
    against one job description.

    Returns normalized scores from 0 to 100.
    """

    job_tokens = tokenize(job_description)

    resume_tokens = [
        tokenize(resume)
        for resume in resumes
    ]

    if not job_tokens or not resume_tokens:
        return [0.0] * len(resumes)

    # Build BM25 corpus using ALL resumes
    bm25 = BM25Okapi(resume_tokens)

    # Calculate raw scores
    raw_scores = bm25.get_scores(job_tokens)

    raw_scores = np.array(raw_scores, dtype=float)

    # Shift scores if negative values exist
    min_score = raw_scores.min()

    if min_score < 0:
        raw_scores = raw_scores - min_score

    # Normalize 0–100
    max_score = raw_scores.max()

    if max_score == 0:
        return [0.0] * len(resumes)

    normalized_scores = (
        raw_scores / max_score
    ) * 100

    return [
        round(float(score), 2)
        for score in normalized_scores
    ]