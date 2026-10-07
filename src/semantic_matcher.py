import os

# Quiet transformers' architecture-registration log spam (the
# "[ERROR] ... not documented" lines) — this is cosmetic noise from
# importing transformers, not an actual error in your pipeline.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. "
        "Install it with: pip install sentence-transformers"
    )

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# Lightweight and strong general-purpose model
MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading semantic model (first run only)...")
def get_semantic_model() -> "SentenceTransformer":
    """
    Load the sentence-transformer model exactly once per Streamlit
    session/process instead of on every import or rerun.

    local_files_only is tried first: once the model is cached on disk
    (~/.cache/torch/sentence_transformers/... or the HF cache dir),
    this skips the network round-trip to Hugging Face Hub that was
    causing the "unauthenticated requests" warning and slowing down
    every load. If it's not cached yet, we fall back to a normal
    (networked) load, which will populate that cache for next time.
    """
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        return SentenceTransformer(MODEL_NAME)


def calculate_semantic_scores(
    job_description: str,
    resumes: list[str]
) -> list[float]:
    """
    Calculate semantic similarity between
    job description and multiple resumes.

    Returns scores from 0 to 100.
    """

    model = get_semantic_model()

    # Encode job description
    job_embedding = model.encode(
        [job_description],
        normalize_embeddings=True
    )

    # Encode resumes
    resume_embeddings = model.encode(
        resumes,
        normalize_embeddings=True
    )

    # Cosine similarity
    similarities = cosine_similarity(
        job_embedding,
        resume_embeddings
    )[0]

    # Convert similarity to 0-100
    scores = []

    for similarity in similarities:

        # Cosine similarity can theoretically be -1 to 1
        score = ((float(similarity) + 1) / 2) * 100

        scores.append(
            round(score, 2)
        )

    return scores