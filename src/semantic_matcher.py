import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def get_vectorizer():
    """
    Lightweight text vectorizer.
    Uses TF-IDF instead of Sentence Transformers so the
    application can run within Render's 512 MB memory limit.
    """
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )


def calculate_semantic_scores(
    job_description: str,
    resumes: list[str]
) -> list[float]:
    """
    Calculate text similarity between a job description
    and multiple resumes.

    Returns scores from 0 to 100.
    """

    if not job_description or not resumes:
        return [0.0] * len(resumes)

    # Combine job description and resumes
    documents = [job_description] + resumes

    vectorizer = get_vectorizer()

    # Convert text into TF-IDF vectors
    embeddings = vectorizer.fit_transform(documents)

    # Job description vector
    job_vector = embeddings[0:1]

    # Resume vectors
    resume_vectors = embeddings[1:]

    # Calculate cosine similarity
    similarities = cosine_similarity(
        job_vector,
        resume_vectors
    )[0]

    # Convert 0-1 similarity into 0-100 score
    scores = [
        round(float(similarity) * 100, 2)
        for similarity in similarities
    ]

    return scores