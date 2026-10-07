import re


def clean_text(text: str) -> str:
    """
    Clean extracted resume/job-description text
    for keyword and semantic matching.
    """

    # Convert to lowercase
    text = text.lower()

    # Replace line breaks and tabs with spaces
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Remove unwanted special characters
    text = re.sub(r"[^a-z0-9+#.\-/ ]", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()