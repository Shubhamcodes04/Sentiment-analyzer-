import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize VADER once (expensive to reload)
vader = SentimentIntensityAnalyzer()


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation rules."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def vader_label(compound: float) -> str:
    """Convert VADER compound score to a human-readable label."""
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def textblob_label(polarity: float) -> str:
    """Convert TextBlob polarity to a human-readable label."""
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"


def subjectivity_label(score: float) -> str:
    """Describe how subjective/objective the text is."""
    if score < 0.3:
        return "Very Objective"
    elif score < 0.5:
        return "Mostly Objective"
    elif score < 0.7:
        return "Mostly Subjective"
    else:
        return "Highly Subjective"


def intensity_label(compound: float) -> str:
    """Describe the emotional intensity."""
    abs_score = abs(compound)
    if abs_score >= 0.75:
        return "Very Strong"
    elif abs_score >= 0.5:
        return "Strong"
    elif abs_score >= 0.25:
        return "Moderate"
    elif abs_score >= 0.05:
        return "Mild"
    else:
        return "Neutral / Flat"


# ── Main Analysis ─────────────────────────────────────────────────────────────

def analyze_text(text: str) -> dict:
    """
    Run full sentiment analysis on input text using VADER + TextBlob.
    Returns a dictionary with all scores, labels, and per-sentence breakdown.
    """

    # ── Overall scores ────────────────────────────────────────────────────
    vader_scores = vader.polarity_scores(text)
    blob = TextBlob(text)

    vader_compound  = vader_scores["compound"]
    textblob_polarity    = round(blob.sentiment.polarity, 4)
    textblob_subjectivity = round(blob.sentiment.subjectivity, 4)

    # Normalize TextBlob polarity (-1 to 1) → (-1 to 1) already matches VADER
    # Average compound for "consensus" score
    consensus_score = round((vader_compound + textblob_polarity) / 2, 4)

    # ── Sentence-level breakdown ──────────────────────────────────────────
    sentences = split_sentences(text)
    sentence_results = []

    for sent in sentences:
        v = vader.polarity_scores(sent)
        b = TextBlob(sent)
        compound = v["compound"]
        polarity = round(b.sentiment.polarity, 3)

        sentence_results.append({
            "sentence": sent,
            "vader_compound": compound,
            "vader_label": vader_label(compound),
            "textblob_polarity": polarity,
            "textblob_label": textblob_label(polarity),
        })

    # ── Final verdict: majority vote ──────────────────────────────────────
    v_label = vader_label(vader_compound)
    t_label = textblob_label(textblob_polarity)

    if v_label == t_label:
        final_verdict = v_label          # both agree
    elif v_label == "Neutral":
        final_verdict = t_label          # TextBlob has a stronger signal
    elif t_label == "Neutral":
        final_verdict = v_label          # VADER has a stronger signal
    else:
        # They disagree — pick the one with stronger signal
        final_verdict = v_label if abs(vader_compound) >= abs(textblob_polarity) else t_label

    return {
        # Overall
        "final_verdict": final_verdict,
        "consensus_score": consensus_score,

        # VADER
        "vader_compound": vader_compound,
        "vader_positive": round(vader_scores["pos"], 4),
        "vader_negative": round(vader_scores["neg"], 4),
        "vader_neutral":  round(vader_scores["neu"], 4),
        "vader_label": v_label,

        # TextBlob
        "textblob_polarity": textblob_polarity,
        "textblob_subjectivity": textblob_subjectivity,
        "textblob_label": t_label,

        # Descriptors
        "intensity": intensity_label(consensus_score),
        "subjectivity_desc": subjectivity_label(textblob_subjectivity),

        # Sentence breakdown
        "sentences": sentence_results,
        "word_count": len(text.split()),
        "sentence_count": len(sentences),
    }
