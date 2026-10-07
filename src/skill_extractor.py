import re


SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++"],
    "c": ["c"],

    "machine learning": [
        "machine learning",
        "machine-learning",
        "ml"
    ],

    "deep learning": [
        "deep learning",
        "deep-learning",
        "dl"
    ],

    "scikit-learn": [
        "scikit-learn",
        "sklearn"
    ],

    "tensorflow": [
        "tensorflow"
    ],

    "pytorch": [
        "pytorch"
    ],

    "xgboost": [
        "xgboost"
    ],

    "nlp": [
        "nlp",
        "natural language processing"
    ],

    "llms": [
        "llm",
        "llms",
        "large language model",
        "large language models"
    ],

    "langchain": [
        "langchain"
    ],

    "rag": [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation"
    ],

    "hugging face": [
        "hugging face",
        "huggingface"
    ],

    "faiss": [
        "faiss"
    ],

    "chromadb": [
        "chromadb",
        "chroma db"
    ],

    "sentence transformers": [
        "sentence transformers",
        "sentence-transformers"
    ],

    "bm25": [
        "bm25"
    ],

    "pandas": [
        "pandas"
    ],

    "numpy": [
        "numpy"
    ],

    "matplotlib": [
        "matplotlib"
    ],

    "seaborn": [
        "seaborn"
    ],

    "flask": [
        "flask"
    ],

    "fastapi": [
        "fastapi"
    ],

    "sql": [
        "sql"
    ],

    "git": [
        "git"
    ],

    "github": [
        "github"
    ],

    "docker": [
        "docker"
    ],

    "aws": [
        "aws",
        "amazon web services"
    ],

    "rest api": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis"
    ],

    "feature engineering": [
        "feature engineering"
    ],

    "hyperparameter tuning": [
        "hyperparameter tuning",
        "hyperparameter optimization"
    ],

    "model evaluation": [
        "model evaluation",
        "model evaluation reports"
    ]
}


def extract_skills(text: str) -> set:
    """
    Extract normalized skills from text.
    """

    text = text.lower()

    found_skills = set()

    for skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            pattern = rf"(?<![a-z0-9+#]){re.escape(alias)}(?![a-z0-9+#])"

            if re.search(pattern, text):
                found_skills.add(skill)
                break

    return found_skills