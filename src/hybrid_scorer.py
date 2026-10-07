def calculate_hybrid_score(
    bm25_score: float,
    semantic_score: float,
    skill_score: float,
    profile_score: float
) -> float:
    """
    Calculate final recommendation score.

    BM25      = 20%
    Semantic  = 30%
    Skills    = 25%
    Profile   = 25%
    """

    final_score = (
        (bm25_score * 0.20)
        + (semantic_score * 0.30)
        + (skill_score * 0.25)
        + (profile_score * 0.25)
    )

    return round(
        final_score,
        2
    )