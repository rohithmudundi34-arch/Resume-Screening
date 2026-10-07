import gc
import html
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import warnings

# ============================================================
# RUNTIME CONFIGURATION
# ============================================================

os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.utils").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text
from src.keyword_matcher import calculate_bm25_scores
from src.semantic_matcher import calculate_semantic_scores
from src.skill_matcher import calculate_skill_match
from src.profile_matcher import calculate_profile_match
from src.hybrid_scorer import calculate_hybrid_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Recruitment Intelligence",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# PREMIUM DARK UI
# ============================================================
# NOTE: <style> is one of the few tags Markdown's HTML-block rules treat
# as "raw until the closing tag", so blank lines/indentation inside it
# are safe here.

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(37,99,235,0.10), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(124,58,237,0.08), transparent 25%),
            #080b11;
        color: #e8ebf2;
    }

    section[data-testid="stSidebar"] {
        background: #0d111a;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1.5px;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 22px;
    }

    .badge-row {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 30px;
    }

    .badge {
        padding: 7px 13px;
        border-radius: 8px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.09);
        color: #cbd5e1;
        font-size: 12px;
        font-weight: 600;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 30px;
        margin-bottom: 14px;
        padding-bottom: 9px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    .panel {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 22px;
    }

    .hero-card {
        padding: 30px;
        border-radius: 17px;
        background:
            linear-gradient(
                135deg,
                rgba(37,99,235,0.18),
                rgba(124,58,237,0.08)
            );
        border: 1px solid rgba(59,130,246,0.30);
        text-align: center;
        margin: 20px 0;
    }

    .hero-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 11px;
        color: #93c5fd;
        font-weight: 700;
    }

    .hero-name {
        font-size: 30px;
        font-weight: 800;
        color: white;
        margin: 8px 0;
    }

    .hero-score {
        font-size: 50px;
        font-weight: 800;
        color: #60a5fa;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 15px 0;
    }

    .stat-card {
        text-align: center;
        padding: 17px;
        border-radius: 12px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .stat-value {
        font-size: 25px;
        font-weight: 800;
        color: #60a5fa;
    }

    .stat-label {
        margin-top: 4px;
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .readiness-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 10px 0 20px 0;
    }

    .readiness-card {
        padding: 16px 18px;
        border-radius: 12px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .readiness-label {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .readiness-value {
        font-size: 22px;
        font-weight: 800;
        line-height: 1.25;
        white-space: normal;
        overflow-wrap: break-word;
        color: #60a5fa;
    }

    .readiness-value.good {
        color: #34d399;
    }

    .readiness-value.ok {
        color: #60a5fa;
    }

    .readiness-value.warn {
        color: #fbbf24;
    }

    .readiness-value.bad {
        color: #f87171;
    }

    @media(max-width: 900px) {
        .readiness-grid {
            grid-template-columns: 1fr;
        }
    }

    .interview-card {
        background: linear-gradient(
            135deg,
            rgba(16,185,129,0.10),
            rgba(59,130,246,0.06)
        );
        border: 1px solid rgba(16,185,129,0.22);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .interview-title {
        color: #6ee7b7;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .question {
        color: #f1f5f9;
        font-weight: 600;
        line-height: 1.6;
    }

    .skill-gap {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px;
        border-radius: 7px;
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.25);
        color: #fca5a5;
        font-size: 12px;
        font-weight: 600;
    }

    .strength {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px;
        border-radius: 7px;
        background: rgba(16,185,129,0.10);
        border: 1px solid rgba(16,185,129,0.25);
        color: #6ee7b7;
        font-size: 12px;
        font-weight: 600;
    }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }

    @media(max-width: 900px) {
        .stat-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    /* ========================================================
       SIDEBAR TEXT
       ======================================================== */
    /* Without this, sidebar text falls back to Streamlit's default
       (near-black) color, which is almost invisible against the dark
       sidebar background above. */

    section[data-testid="stSidebar"] * {
        color: #d5dcef !important;
    }

    /* ========================================================
       TEXT AREA (Job Description input)
       ======================================================== */
    /* Without this, the textarea keeps Streamlit's default light
       theme (white box, dark text) instead of matching the app. */

    div[data-testid="stTextArea"] textarea {
        background-color: #0d111a !important;
        color: #f1f5f9 !important;
        caret-color: #f1f5f9 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }

    /* ========================================================
       FILE UPLOADER (Upload Resumes)
       ======================================================== */
    /* Same issue as the textarea - force the drag-and-drop box and
       its icon/text to match the dark theme instead of Streamlit's
       default white uploader. */

    [data-testid*="FileUploader"] {
        background-color: #0d111a !important;
        color: #f1f5f9 !important;
        border-color: rgba(255,255,255,0.16) !important;
    }

    [data-testid*="FileUploader"] section {
        background-color: #0d111a !important;
        border: 1px dashed rgba(255,255,255,0.22) !important;
        border-radius: 10px !important;
    }

    [data-testid*="FileUploader"] * {
        color: #f1f5f9 !important;
        background-color: transparent !important;
    }

    [data-testid*="FileUploader"] svg {
        fill: #f1f5f9 !important;
        stroke: #f1f5f9 !important;
    }

    [data-testid*="FileUploader"] small {
        color: #94a3b8 !important;
    }

    [data-testid*="FileUploader"] button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def flatten_html(raw: str) -> str:
    """Collapse a hand-indented, multi-line HTML template into one line.

    Streamlit passes markdown-with-html through a Markdown parser before
    rendering. Any blank line inside a raw HTML block ends that block for
    the parser, and any indented (4+ spaces) line that follows then gets
    treated as a *fenced code block* instead of HTML - printing literal
    tags on the page instead of rendering them. Stripping per-line
    whitespace and blank lines avoids that entirely.
    """
    lines = [line.strip() for line in raw.splitlines()]
    return " ".join(line for line in lines if line)


def clamp_pct(value):
    try:
        return max(0.0, min(round(float(value), 1), 100.0))
    except (TypeError, ValueError):
        return 0.0


def get_initials(name):
    base = os.path.splitext(str(name))[0]

    tokens = [
        x for x in
        base.replace("_", " ")
            .replace("-", " ")
            .replace(".", " ")
            .split()
        if x
    ]

    if not tokens:
        return "NA"

    if len(tokens) == 1:
        return tokens[0][:2].upper()

    return (tokens[0][0] + tokens[1][0]).upper()


def render_skillbar(label, value):

    pct = clamp_pct(value)
    safe_label = html.escape(str(label))

    st.markdown(
        flatten_html(
            f"""
            <div style="margin-bottom:16px">
                <div style="display:flex; justify-content:space-between;
                            color:#cbd5e1; font-size:13px; font-weight:600;
                            margin-bottom:6px;">
                    <span>{safe_label}</span>
                    <span>{pct}%</span>
                </div>
                <div style="height:8px; border-radius:999px;
                            background:rgba(255,255,255,.07); overflow:hidden;">
                    <div style="width:{pct}%; height:100%; background:#3b82f6;
                                border-radius:999px;"></div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# INTERVIEW INTELLIGENCE
# ============================================================

def generate_interview_questions(candidate, job_description):
    """Build a short list of personalized questions from the candidate's scores.

    `job_description` is accepted so this can be extended later (e.g. to
    pull specific requirements into the question text) without changing
    the call sites.
    """

    questions = []

    skill_score = candidate["Skills"]
    semantic = candidate["Semantic"]
    role = candidate["Role"]
    experience = candidate["Experience"]

    if skill_score >= 80:
        questions.append(
            "You have a strong skill match. Explain one project where "
            "you applied the core technologies required for this role."
        )
    else:
        questions.append(
            "Which technical skills from this role are you most "
            "confident in, and where have you applied them?"
        )

    if semantic >= 75:
        questions.append(
            "Walk through a project from your resume that is most "
            "relevant to this job description. What was your contribution?"
        )
    else:
        questions.append(
            "Which experience from your background do you believe is "
            "most transferable to this position?"
        )

    if role >= 75:
        questions.append(
            "Describe a situation where you solved a difficult problem "
            "related to the responsibilities of this role."
        )
    else:
        questions.append(
            "How would you approach learning the responsibilities of "
            "this role during your first 90 days?"
        )

    if experience >= 75:
        questions.append(
            "Tell me about the most technically challenging problem "
            "you solved in your previous experience."
        )
    else:
        questions.append(
            "Describe a project that demonstrates your ability to "
            "perform at the level expected for this position."
        )

    questions.append(
        "Explain one technical decision you made in a project and why "
        "you chose that approach over alternatives."
    )

    questions.append(
        "If your solution failed in production, how would you "
        "investigate the issue and prevent it from happening again?"
    )

    return questions


def render_interview_intelligence(candidate, job_description):

    st.markdown(
        '<div class="section-title">Interview Intelligence</div>',
        unsafe_allow_html=True,
    )

    score = candidate["Final Score"]

    if score >= 85:
        readiness = "Highly Interview Ready"
        readiness_class = "good"
    elif score >= 70:
        readiness = "Interview Recommended"
        readiness_class = "ok"
    elif score >= 55:
        readiness = "Needs Technical Validation"
        readiness_class = "warn"
    else:
        readiness = "Low Match — Further Review Required"
        readiness_class = "bad"

    st.markdown(
        flatten_html(
            f"""
            <div class="readiness-grid">
                <div class="readiness-card">
                    <div class="readiness-label">Interview Readiness</div>
                    <div class="readiness-value {readiness_class}">{html.escape(readiness)}</div>
                </div>
                <div class="readiness-card">
                    <div class="readiness-label">Candidate Match</div>
                    <div class="readiness-value">{score}%</div>
                </div>
                <div class="readiness-card">
                    <div class="readiness-label">Experience</div>
                    <div class="readiness-value">{candidate['Candidate Experience']} yrs</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # STRENGTHS
    # --------------------------------------------------------

    st.markdown("#### Candidate Strengths")

    strengths = []

    if candidate["BM25"] >= 75:
        strengths.append("Strong keyword alignment")

    if candidate["Semantic"] >= 75:
        strengths.append("Strong semantic relevance")

    if candidate["Skills"] >= 75:
        strengths.append("Strong technical skill match")

    if candidate["Profile"] >= 75:
        strengths.append("Strong profile alignment")

    if candidate["Education"] >= 75:
        strengths.append("Education alignment")

    if candidate["Role"] >= 75:
        strengths.append("Strong role fit")

    if candidate["Experience"] >= 75:
        strengths.append("Relevant experience")

    if not strengths:
        strengths.append("Candidate requires deeper manual evaluation")

    strength_chips = "".join(
        f'<span class="strength">{html.escape(item)}</span>'
        for item in strengths
    )

    st.markdown(strength_chips, unsafe_allow_html=True)

    # --------------------------------------------------------
    # AREAS TO VALIDATE
    # --------------------------------------------------------

    st.markdown("#### Areas to Validate")

    gaps = []

    if candidate["Skills"] < 70:
        gaps.append("Technical skills")

    if candidate["Semantic"] < 70:
        gaps.append("Resume relevance")

    if candidate["Role"] < 70:
        gaps.append("Role alignment")

    if candidate["Experience"] < 70:
        gaps.append("Experience depth")

    if candidate["Education"] < 70:
        gaps.append("Education alignment")

    if not gaps:
        gaps.append("No major automated gaps detected")

    gap_chips = "".join(
        f'<span class="skill-gap">{html.escape(item)}</span>'
        for item in gaps
    )

    st.markdown(gap_chips, unsafe_allow_html=True)

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    st.markdown("#### Personalized Interview Questions")

    questions = generate_interview_questions(
        candidate,
        job_description
    )

    for index, question in enumerate(questions, start=1):

        st.markdown(
            flatten_html(
                f"""
                <div class="interview-card">
                    <div class="interview-title">Question {index}</div>
                    <div class="question">{html.escape(question)}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )




# ============================================================
# PDF EXTRACTION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
    ttl=3600
)
def extract_and_clean_resume(file_bytes, filename):

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        raw_text = extract_text_from_pdf(temp_path)

        if not raw_text or not raw_text.strip():

            return (
                "",
                f"No extractable text found in '{filename}'."
            )

        return clean_text(raw_text), None

    except Exception as exc:

        return (
            "",
            f"Failed to read '{filename}': {exc}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def extract_all_resumes(files):

    n = len(files)

    ordered_text = [None] * n
    ordered_error = [None] * n

    progress = st.progress(0)
    status = st.empty()

    completed = 0

    with ThreadPoolExecutor(
        max_workers=min(3, max(1, n))
    ) as pool:

        future_to_index = {
            pool.submit(
                extract_and_clean_resume,
                f.getvalue(),
                f.name
            ): idx
            for idx, f in enumerate(files)
        }

        for future in as_completed(future_to_index):

            idx = future_to_index[future]

            text, error = future.result()

            ordered_text[idx] = text
            ordered_error[idx] = error

            completed += 1

            status.write(
                f"Processing resumes: {completed}/{n}"
            )

            progress.progress(
                completed / n
            )

    status.write("Resume processing completed.")

    resumes = []
    names = []
    errors = []

    for idx, f in enumerate(files):

        if ordered_error[idx]:

            errors.append(
                ordered_error[idx]
            )

        else:

            resumes.append(
                ordered_text[idx]
            )

            names.append(
                f.name
            )

    return resumes, names, errors


# ============================================================
# SCORING
# ============================================================

def _score_one_resume(job_description, resume):

    skill_result = calculate_skill_match(
        job_description,
        resume
    )

    profile_result = calculate_profile_match(
        job_description,
        resume
    )

    return {
        "skill_score": skill_result["score"],
        "profile": profile_result,
    }


@st.cache_data(
    show_spinner=False,
    max_entries=10,
    ttl=3600
)
def score_candidates(
    job_description,
    resumes
):

    bm25_scores = calculate_bm25_scores(
        job_description,
        list(resumes)
    )

    semantic_scores = calculate_semantic_scores(
        job_description,
        list(resumes)
    )

    skill_scores = [None] * len(resumes)
    profile_results = [None] * len(resumes)

    with ThreadPoolExecutor(
        max_workers=min(3, max(1, len(resumes)))
    ) as pool:

        future_to_index = {
            pool.submit(
                _score_one_resume,
                job_description,
                resume
            ): idx
            for idx, resume in enumerate(resumes)
        }

        for future in as_completed(
            future_to_index
        ):

            idx = future_to_index[future]

            result = future.result()

            skill_scores[idx] = result["skill_score"]
            profile_results[idx] = result["profile"]

    return {
        "bm25": bm25_scores,
        "semantic": semantic_scores,
        "skills": skill_scores,
        "profiles": profile_results,
    }


# ============================================================
# ANALYSIS
# ============================================================

def run_analysis(
    job_description_input,
    uploaded_files
):

    job_description = clean_text(
        job_description_input
    )

    resumes, names, errors = extract_all_resumes(
        uploaded_files
    )

    if not resumes:

        st.session_state["analysis_error"] = (
            "No resumes could be processed."
        )

        st.session_state.pop("results", None)

        return

    with st.spinner(
        "Running AI recruitment intelligence engine..."
    ):

        scores = score_candidates(
            job_description,
            tuple(resumes)
        )

    results = []

    for i, name in enumerate(names):

        profile = scores["profiles"][i]

        final_score = calculate_hybrid_score(
            scores["bm25"][i],
            scores["semantic"][i],
            scores["skills"][i],
            profile["score"],
        )

        results.append(
            {
                "Candidate": name,

                "BM25": clamp_pct(
                    scores["bm25"][i]
                ),

                "Semantic": clamp_pct(
                    scores["semantic"][i]
                ),

                "Skills": clamp_pct(
                    scores["skills"][i]
                ),

                "Profile": clamp_pct(
                    profile["score"]
                ),

                "Education": clamp_pct(
                    profile["education_score"]
                ),

                "Role": clamp_pct(
                    profile["role_score"]
                ),

                "Experience": clamp_pct(
                    profile["experience_score"]
                ),

                "Required Experience":
                    profile["required_years"],

                "Candidate Experience":
                    profile["candidate_years"],

                "Final Score":
                    clamp_pct(final_score),
            }
        )

    results.sort(
        key=lambda x: x["Final Score"],
        reverse=True
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        result["Rank"] = index

    st.session_state["results"] = results

    st.session_state[
        "extraction_errors"
    ] = errors

    st.session_state[
        "job_description"
    ] = job_description

    st.session_state.pop(
        "analysis_error",
        None
    )

    del resumes
    del scores

    gc.collect()


# ============================================================
# RADAR
# ============================================================

def render_radar_chart(results):

    dimensions = [
        "BM25",
        "Semantic",
        "Skills",
        "Profile",
        "Education",
        "Role",
        "Experience",
    ]

    fig = go.Figure()

    palette = [
        "#3b82f6",
        "#f472b6",
        "#fbbf24",
        "#34d399",
        "#a78bfa",
    ]

    for i, candidate in enumerate(results[:5]):

        values = [
            candidate[x]
            for x in dimensions
        ]

        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=dimensions + [dimensions[0]],
                fill="toself",
                name=candidate["Candidate"],
                opacity=.25,
                line=dict(
                    color=palette[
                        i % len(palette)
                    ],
                    width=2
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(255,255,255,.10)"
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,.10)"
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# RESULTS
# ============================================================

def render_results(
    results,
    extraction_errors
):

    if extraction_errors:

        with st.expander(
            f"{len(extraction_errors)} resume(s) skipped",
            expanded=False,
        ):

            for error in extraction_errors:

                st.warning(html.escape(error))

    winner = results[0]

    # --------------------------------------------------------
    # WINNER
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">AI Recommendation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        flatten_html(
            f"""
            <div class="hero-card">
                <div class="hero-label">Top Candidate</div>
                <div class="hero-name">{html.escape(winner["Candidate"])}</div>
                <div class="hero-score">{winner["Final Score"]}%</div>
                <div style="color:#94a3b8">Hybrid Candidate Match Score</div>
            </div>
            """
        ),
        unsafe_allow_html=True
    )

    avg = round(
        sum(
            x["Final Score"]
            for x in results
        ) / len(results),
        1
    )

    lead = round(
        winner["Final Score"] -
        (
            results[1]["Final Score"]
            if len(results) > 1
            else 0
        ),
        1
    )

    st.markdown(
        flatten_html(
            f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(results)}</div>
                    <div class="stat-label">Candidates</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{winner["Final Score"]}%</div>
                    <div class="stat-label">Top Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg}%</div>
                    <div class="stat-label">Pool Average</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">+{lead}</div>
                    <div class="stat-label">Lead Over #2</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "Candidate Analysis",
            "Candidate Comparison",
            "Interview Intelligence",
        ]
    )

    # ========================================================
    # CANDIDATE ANALYSIS
    # ========================================================

    with tab1:

        st.markdown(
            '<div class="section-title">Winner Score Breakdown</div>',
            unsafe_allow_html=True
        )

        metrics = [
            ("BM25 Keyword Match", winner["BM25"]),
            ("Semantic Similarity", winner["Semantic"]),
            ("Skill Match", winner["Skills"]),
            ("Profile Match", winner["Profile"]),
            ("Education", winner["Education"]),
            ("Role Fit", winner["Role"]),
            ("Experience", winner["Experience"]),
        ]

        c1, c2 = st.columns(2)

        for index, (label, value) in enumerate(metrics):

            with c1 if index % 2 == 0 else c2:

                render_skillbar(
                    label,
                    value
                )

        st.markdown(
            '<div class="section-title">Candidate Radar</div>',
            unsafe_allow_html=True
        )

        render_radar_chart(results)

    # ========================================================
    # COMPARISON
    # ========================================================

    with tab2:

        st.markdown(
            '<div class="section-title">Candidate Ranking</div>',
            unsafe_allow_html=True
        )

        dataframe = pd.DataFrame(results)

        columns = [
            "Rank",
            "Candidate",
            "Final Score",
            "BM25",
            "Semantic",
            "Skills",
            "Profile",
            "Education",
            "Role",
            "Experience",
            "Required Experience",
            "Candidate Experience",
        ]

        st.dataframe(
            dataframe[columns],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = dataframe[columns].to_csv(index=False)

        st.download_button(
            "Download Candidate Analysis (CSV)",
            data=csv_data,
            file_name="ai_candidate_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown(
            '<div class="section-title">Detailed Candidates</div>',
            unsafe_allow_html=True
        )

        for candidate in results:

            with st.expander(
                f'Rank {candidate["Rank"]} — '
                f'{candidate["Candidate"]} — '
                f'{candidate["Final Score"]}%'
            ):

                a, b, c, d = st.columns(4)

                with a:
                    st.metric(
                        "Final Score",
                        f'{candidate["Final Score"]}%'
                    )

                with b:
                    st.metric(
                        "Skills",
                        f'{candidate["Skills"]}%'
                    )

                with c:
                    st.metric(
                        "Role Fit",
                        f'{candidate["Role"]}%'
                    )

                with d:
                    st.metric(
                        "Experience",
                        f'{candidate["Candidate Experience"]} yrs'
                    )

    # ========================================================
    # INTERVIEW INTELLIGENCE
    # ========================================================

    with tab3:

        candidate_names = [c["Candidate"] for c in results]

        selected_name = st.selectbox(
            "Choose a candidate to prepare an interview plan for",
            options=candidate_names,
            index=0,
        )

        selected_candidate = next(
            c for c in results if c["Candidate"] == selected_name
        )

        render_interview_intelligence(
            selected_candidate,
            st.session_state.get(
                "job_description",
                ""
            )
        )

    # --------------------------------------------------------
    # RESTART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Start Over</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "Start New Screening",
        use_container_width=True
    ):

        for key in [
            "results",
            "extraction_errors",
            "analysis_error",
            "job_description",
        ]:

            st.session_state.pop(
                key,
                None
            )

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">AI Recruitment Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    flatten_html(
        """
        <div class="subtitle">
            Intelligent Resume Screening • Candidate Ranking •
            Interview Intelligence
        </div>
        """
    ),
    unsafe_allow_html=True
)

st.markdown(
    flatten_html(
        """
        <div class="badge-row">
            <span class="badge">BM25 Keyword Match</span>
            <span class="badge">Semantic Similarity</span>
            <span class="badge">Skill Matching</span>
            <span class="badge">Profile Matching</span>
            <span class="badge">Education</span>
            <span class="badge">Role Fit</span>
            <span class="badge">Experience</span>
            <span class="badge">Interview Intelligence</span>
        </div>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### Recruitment Settings")

    st.info(
        "Upload up to 10 resumes. "
        "Candidates are ranked using a hybrid AI scoring pipeline."
    )

    st.markdown("---")

    st.markdown("### Intelligence Modules")

    modules = [
        "Keyword Matching",
        "Semantic Matching",
        "Skill Analysis",
        "Profile Analysis",
        "Education Analysis",
        "Role Fit",
        "Experience Analysis",
        "Interview Readiness",
        "Personalized Questions",
    ]

    for module in modules:
        st.write(module)


# ============================================================
# SHOW EXISTING RESULTS
# ============================================================

if st.session_state.get("results"):

    render_results(
        st.session_state["results"],
        st.session_state.get(
            "extraction_errors",
            []
        )
    )

    st.stop()


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.markdown(
    '<div class="section-title">Job Description</div>',
    unsafe_allow_html=True
)

job_description_input = st.text_area(
    "Paste the job description here",
    height=250,
    placeholder=(
        "Example:\n\n"
        "We are looking for an AI/ML Engineer "
        "with Python, Machine Learning, NLP, "
        "TensorFlow, LangChain and GenAI experience..."
    ),
)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload Resumes</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Upload candidate resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:

    if len(uploaded_files) > 10:

        st.error(
            "Maximum 10 resumes can be uploaded."
        )

        st.stop()

    st.success(
        f"{len(uploaded_files)} resume(s) ready for screening."
    )


# ============================================================
# ANALYZE
# ============================================================

analyze = st.button(
    "Analyze & Prepare Interview Plan",
    use_container_width=True,
    type="primary"
)


# ============================================================
# ERROR
# ============================================================

if st.session_state.get(
    "analysis_error"
):

    st.error(
        st.session_state[
            "analysis_error"
        ]
    )


# ============================================================
# RUN
# ============================================================

if analyze:

    if not job_description_input.strip():

        st.error(
            "Please enter a job description."
        )

        st.stop()

    if not uploaded_files:

        st.error(
            "Please upload at least one resume."
        )

        st.stop()

    if len(uploaded_files) > 10:

        st.error(
            "Maximum 10 resumes are allowed."
        )

        st.stop()

    run_analysis(
        job_description_input,
        uploaded_files
    )

    st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        flatten_html(
            """
            <div class="panel" style="text-align:center;margin-top:20px">
                <div style="font-size:18px; font-weight:700; color:#f8fafc;
                            margin-bottom:8px;">
                    Build an Interview-Ready Candidate Shortlist
                </div>
                <div style="font-size:14px; color:#94a3b8; line-height:1.7;">
                    Paste a job description and upload candidate resumes.
                    The system will rank candidates using hybrid AI matching,
                    identify strengths and validation areas, and generate
                    personalized technical interview questions for whichever
                    candidate you select.
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True
    )