import re
from datetime import datetime


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for matching.
    """

    if not text:
        return ""

    text = text.lower()

    # Normalize dash characters
    text = (
        text
        .replace("\u2013", "-")   # en dash
        .replace("\u2014", "-")   # em dash
        .replace("\u2212", "-")   # minus sign
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EDUCATION KEYWORDS
# ============================================================

EDUCATION_KEYWORDS = {

    "b.tech": [
        "b.tech",
        "btech",
        "bachelor of technology"
    ],

    "be": [
        "b.e",
        "be",
        "bachelor of engineering"
    ],

    "computer science": [
        "computer science",
        "computer science engineering",
        "cse"
    ],

    "artificial intelligence": [
        "artificial intelligence",
        "ai"
    ],

    "machine learning": [
        "machine learning",
        "ml"
    ],

    "data science": [
        "data science",
        "data scientist"
    ],

    "information technology": [
        "information technology",
        "information technology engineering",
        "it"
    ],

    "electrical engineering": [
        "electrical engineering",
        "eee"
    ],

    "electronics": [
        "electronics",
        "ece"
    ],

    "m.tech": [
        "m.tech",
        "mtech",
        "master of technology"
    ],

    "mca": [
        "mca",
        "master of computer applications"
    ],

    "me": [
        "m.e",
        "master of engineering"
    ]
}


# ============================================================
# ROLE KEYWORDS
# ============================================================

ROLE_KEYWORDS = {

    "ai/ml engineer": [
        "ai/ml engineer",
        "ai ml engineer",
        "ai engineer",
        "ml engineer",
        "artificial intelligence engineer",
        "machine learning engineer"
    ],

    "machine learning": [
        "machine learning",
        "machine learning engineer"
    ],

    "python developer": [
        "python developer",
        "python development"
    ],

    "data scientist": [
        "data scientist",
        "data science"
    ],

    "data analyst": [
        "data analyst",
        "data analysis"
    ],

    "software engineer": [
        "software engineer",
        "software developer"
    ],

    "backend developer": [
        "backend developer",
        "backend engineer"
    ],

    "full stack developer": [
        "full stack developer",
        "fullstack developer"
    ],

    "deep learning engineer": [
        "deep learning engineer",
        "deep learning"
    ],

    "nlp engineer": [
        "nlp engineer",
        "natural language processing"
    ],

    "generative ai engineer": [
        "generative ai engineer",
        "generative ai"
    ]
}


# ============================================================
# EXTRACT EDUCATION
# ============================================================

def extract_education(
    text: str
) -> list[str]:

    text = normalize_text(text)

    found = []

    for education, keywords in EDUCATION_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:

                found.append(
                    education
                )

                break

    return sorted(
        set(found)
    )


# ============================================================
# EXTRACT ROLES
# ============================================================

def extract_roles(
    text: str
) -> list[str]:

    text = normalize_text(text)

    found = []

    for role, keywords in ROLE_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:

                found.append(
                    role
                )

                break

    return sorted(
        set(found)
    )


# ============================================================
# EXTRACT REQUIRED EXPERIENCE FROM JOB
# ============================================================

def extract_required_experience(
    text: str
) -> float:
    """
    Extract minimum required experience.

    Supports:

        2+ years
        minimum 2 years
        at least 2 years
        2 years of experience
        experience of 2 years
        2-4 years
    """

    text = normalize_text(text)

    # --------------------------------------------------------
    # 2+ years
    # --------------------------------------------------------

    patterns = [

        r"(\d+(?:\.\d+)?)\s*\+\s*years?",

        r"minimum\s+(?:of\s+)?"
        r"(\d+(?:\.\d+)?)\s*years?",

        r"at\s+least\s+"
        r"(\d+(?:\.\d+)?)\s*years?",

        r"(\d+(?:\.\d+)?)\s*years?"
        r"\s*(?:of\s+)?experience",

        r"experience\s*(?:of\s+)?"
        r"(\d+(?:\.\d+)?)\s*years?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            return float(
                match.group(1)
            )

    # --------------------------------------------------------
    # Experience range
    # --------------------------------------------------------

    range_pattern = (
        r"(\d+(?:\.\d+)?)"
        r"\s*(?:-|to)"
        r"\s*(\d+(?:\.\d+)?)"
        r"\s*years?"
    )

    match = re.search(
        range_pattern,
        text,
        flags=re.IGNORECASE
    )

    if match:

        # Lower bound = minimum requirement
        return float(
            match.group(1)
        )

    return 0.0


# ============================================================
# MONTH MAP
# ============================================================

MONTH_MAP = {

    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}

# ============================================================
# EXTRACT DATE RANGES
# ============================================================

def extract_date_ranges(
    text: str
) -> list[tuple[int, int]]:
    """
    Extract date ranges from resume text.

    Supports:

        Jun 2025 - Sep 2025
        Jun 2025 – Sep 2025
        Jun 2025 — Sep 2025
        Jun 2025 to Sep 2025

    Also supports cleaned PDF text where the
    separator has disappeared:

        Jun 2025 Sep 2025

    Also supports:

        Jan 2024 - Present
        Jan 2024 - Current
    """

    if not text:
        return []

    # --------------------------------------------------------
    # NORMALIZE TEXT
    # --------------------------------------------------------

    text = text.lower()

    text = (
        text
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # --------------------------------------------------------
    # MONTH PATTERN
    # --------------------------------------------------------

    month_pattern = (
        r"(jan(?:uary)?|"
        r"feb(?:ruary)?|"
        r"mar(?:ch)?|"
        r"apr(?:il)?|"
        r"may|"
        r"jun(?:e)?|"
        r"jul(?:y)?|"
        r"aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|"
        r"nov(?:ember)?|"
        r"dec(?:ember)?)"
    )

    # --------------------------------------------------------
    # DATE PATTERN
    # --------------------------------------------------------

    date_pattern = (
        rf"{month_pattern}"
        r"\s+"
        r"(\d{4})"
    )

    # --------------------------------------------------------
    # RANGE PATTERN
    #
    # IMPORTANT:
    #
    # The separator is OPTIONAL.
    #
    # This is necessary because your clean_text()
    # removes the dash from:
    #
    # Jun 2025 – Sep 2025
    #
    # and produces:
    #
    # jun 2025 sep 2025
    # --------------------------------------------------------

    range_pattern = (
        rf"({date_pattern})"
        r"\s*"
        r"(?:-|to)?"
        r"\s*"
        rf"({date_pattern}|present|current)"
    )

    ranges = []

    # --------------------------------------------------------
    # FIND DATE RANGES
    # --------------------------------------------------------

    for match in re.finditer(
        range_pattern,
        text,
        flags=re.IGNORECASE
    ):

        # ----------------------------------------------------
        # START DATE
        # ----------------------------------------------------

        start_text = match.group(1)

        start_match = re.search(
            date_pattern,
            start_text,
            flags=re.IGNORECASE
        )

        if not start_match:
            continue

        start_month = (
            start_match
            .group(1)
            .lower()[:3]
        )

        start_year = int(
            start_match.group(2)
        )

        if start_month not in MONTH_MAP:
            continue

        start_month_number = (
            MONTH_MAP[start_month]
        )

        # ----------------------------------------------------
        # END DATE
        # ----------------------------------------------------

        end_text = match.group(
            match.lastindex
        ).strip()

        end_text_lower = (
            end_text.lower()
        )

        # ----------------------------------------------------
        # PRESENT / CURRENT
        # ----------------------------------------------------

        if end_text_lower in {
            "present",
            "current"
        }:

            current_date = datetime.now()

            end_month_number = (
                current_date.month
            )

            end_year = (
                current_date.year
            )

        # ----------------------------------------------------
        # NORMAL END DATE
        # ----------------------------------------------------

        else:

            end_match = re.search(
                date_pattern,
                end_text,
                flags=re.IGNORECASE
            )

            if not end_match:
                continue

            end_month = (
                end_match
                .group(1)
                .lower()[:3]
            )

            if end_month not in MONTH_MAP:
                continue

            end_month_number = (
                MONTH_MAP[end_month]
            )

            end_year = int(
                end_match.group(2)
            )

        # ----------------------------------------------------
        # CONVERT TO MONTH INDEX
        # ----------------------------------------------------

        start_index = (
            start_year * 12
            + start_month_number
        )

        end_index = (
            end_year * 12
            + end_month_number
        )

        # ----------------------------------------------------
        # VALID RANGE
        # ----------------------------------------------------

        if end_index >= start_index:

            ranges.append(
                (
                    start_index,
                    end_index
                )
            )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    return sorted(
        set(ranges)
    )
# ============================================================
# MERGE OVERLAPPING DATE RANGES
# ============================================================

def merge_date_ranges(
    ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """
    Merge overlapping employment periods.
    """

    if not ranges:
        return []

    ranges = sorted(
        ranges
    )

    merged = [
        ranges[0]
    ]

    for start, end in ranges[1:]:

        previous_start, previous_end = (
            merged[-1]
        )

        if start <= previous_end + 1:

            merged[-1] = (
                previous_start,
                max(
                    previous_end,
                    end
                )
            )

        else:

            merged.append(
                (
                    start,
                    end
                )
            )

    return merged


# ============================================================
# EXTRACT RESUME EXPERIENCE
# ============================================================

def extract_resume_experience(
    text: str
) -> float:
    """
    Calculate total professional experience
    using explicit experience and employment
    date ranges.
    """

    if not text:
        return 0.0

    text = normalize_text(text)

    total_months = 0.0

    # ========================================================
    # EXPLICIT YEARS
    # ========================================================

    year_matches = re.findall(
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*years?",
        text,
        flags=re.IGNORECASE
    )

    if year_matches:

        values = [
            float(value)
            for value in year_matches
        ]

        total_months = (
            max(values) * 12
        )

    # ========================================================
    # EXPLICIT MONTHS
    # ========================================================

    month_matches = re.findall(
        r"(\d+(?:\.\d+)?)"
        r"\s*months?",
        text,
        flags=re.IGNORECASE
    )

    if month_matches:

        values = [
            float(value)
            for value in month_matches
        ]

        explicit_months = max(
            values
        )

        total_months = max(
            total_months,
            explicit_months
        )

    # ========================================================
    # DATE RANGES
    # ========================================================

    date_ranges = extract_date_ranges(
        text
    )

    if date_ranges:

        merged_ranges = merge_date_ranges(
            date_ranges
        )

        date_range_months = 0

        for start, end in merged_ranges:

            months = (
                end - start + 1
            )

            date_range_months += months

        total_months = max(
            total_months,
            float(date_range_months)
        )

    # ========================================================
    # RETURN EXPERIENCE IN YEARS
    # ========================================================

    return round(
        total_months / 12,
        2
    )
# ============================================================
# CALCULATE EXPERIENCE SCORE
# ============================================================

def calculate_experience_score(
    job_description: str,
    resume_text: str
) -> dict:
    """
    Compare required experience with
    candidate experience.
    """

    required_years = (
        extract_required_experience(
            job_description
        )
    )

    candidate_years = (
        extract_resume_experience(
            resume_text
        )
    )

    # ========================================================
    # NO EXPERIENCE REQUIREMENT
    # ========================================================

    if required_years <= 0:

        score = 100.0

    # ========================================================
    # CANDIDATE MEETS REQUIREMENT
    # ========================================================

    elif candidate_years >= required_years:

        score = 100.0

    # ========================================================
    # PARTIAL MATCH
    # ========================================================

    else:

        score = (
            candidate_years
            / required_years
        ) * 100

    return {

        "required_years": round(
            required_years,
            2
        ),

        "candidate_years": round(
            candidate_years,
            2
        ),

        "score": round(
            min(score, 100.0),
            2
        )
    }


# ============================================================
# EDUCATION MATCH
# ============================================================

def calculate_education_match(
    job_education: list[str],
    resume_education: list[str]
) -> float:
    """
    Calculate education compatibility.
    """

    if not job_education:

        return 100.0

    if not resume_education:

        return 0.0

    matched = 0

    for required in job_education:

        if required in resume_education:

            matched += 1

    return round(
        (
            matched
            / len(job_education)
        ) * 100,
        2
    )


# ============================================================
# ROLE MATCH
# ============================================================

def calculate_role_match(
    job_roles: list[str],
    resume_roles: list[str]
) -> float:
    """
    Calculate role compatibility.
    """

    if not job_roles:

        return 100.0

    if not resume_roles:

        return 0.0

    matched = 0

    for required in job_roles:

        if required in resume_roles:

            matched += 1

    return round(
        (
            matched
            / len(job_roles)
        ) * 100,
        2
    )


# ============================================================
# PROFILE MATCH
# ============================================================

def calculate_profile_match(
    job_description: str,
    resume_text: str
) -> dict:
    """
    Calculate complete profile compatibility.

    Components:

        Education
        Role
        Experience
    """

    job_description = normalize_text(
        job_description
    )

    resume_text = normalize_text(
        resume_text
    )

    # ========================================================
    # EXTRACT JOB INFORMATION
    # ========================================================

    job_education = extract_education(
        job_description
    )

    job_roles = extract_roles(
        job_description
    )

    # ========================================================
    # EXTRACT RESUME INFORMATION
    # ========================================================

    resume_education = extract_education(
        resume_text
    )

    resume_roles = extract_roles(
        resume_text
    )

    # ========================================================
    # EDUCATION
    # ========================================================

    education_score = (
        calculate_education_match(
            job_education,
            resume_education
        )
    )

    # ========================================================
    # ROLE
    # ========================================================

    role_score = (
        calculate_role_match(
            job_roles,
            resume_roles
        )
    )

    # ========================================================
    # EXPERIENCE
    # ========================================================

    experience_result = (
        calculate_experience_score(
            job_description,
            resume_text
        )
    )

    experience_score = (
        experience_result["score"]
    )

    # ========================================================
    # PROFILE WEIGHTING
    # ========================================================

    overall_score = (

        education_score * 0.30

        + role_score * 0.40

        + experience_score * 0.30
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "score": round(
            overall_score,
            2
        ),

        "education_score": round(
            education_score,
            2
        ),

        "role_score": round(
            role_score,
            2
        ),

        "experience_score": round(
            experience_score,
            2
        ),

        "required_years": (
            experience_result[
                "required_years"
            ]
        ),

        "candidate_years": (
            experience_result[
                "candidate_years"
            ]
        ),

        "job_education": (
            job_education
        ),

        "resume_education": (
            resume_education
        ),

        "job_roles": (
            job_roles
        ),

        "resume_roles": (
            resume_roles
        )
    }