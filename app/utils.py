import os
from typing import Dict, Iterable, Optional, Tuple

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")

SENTIMENT_MAP = {
    "recommended": "positive",
    "positive": "positive",
    "not_recommended": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "no_idea": "unknown",
}

SENTIMENT_DISPLAY = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "unknown": "Unknown",
}


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def get_professor_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(df, ["professor_name_raw", "teacher_name", "professor_name"])


def get_cluster_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(df, ["Cluster_Agg", "Cluster_Label", "cluster"])


def get_comment_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(df, ["comment_text", "review_text", "text"])


def get_sentiment_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(df, ["sentiment_label", "label"])


def _normalize_sentiment(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    key = str(value).strip().lower()
    return SENTIMENT_MAP.get(key, "unknown")


def sentiment_counts(df: pd.DataFrame) -> pd.Series:
    sentiment_col = get_sentiment_column(df)
    if sentiment_col is None:
        return pd.Series(dtype=int)
    normalized = df[sentiment_col].map(_normalize_sentiment)
    counts = normalized.value_counts()
    ordered = [
        SENTIMENT_DISPLAY["positive"],
        SENTIMENT_DISPLAY["negative"],
        SENTIMENT_DISPLAY["neutral"],
        SENTIMENT_DISPLAY["unknown"],
    ]
    display_counts = pd.Series(
        {
            SENTIMENT_DISPLAY[key]: int(counts.get(key, 0))
            for key in ["positive", "negative", "neutral", "unknown"]
        }
    )
    return display_counts.reindex(ordered).fillna(0).astype(int)


def sentiment_pos_neg_counts(df: pd.DataFrame) -> Tuple[int, int]:
    sentiment_col = get_sentiment_column(df)
    if sentiment_col is None:
        return 0, 0
    normalized = df[sentiment_col].map(_normalize_sentiment)
    return int((normalized == "positive").sum()), int((normalized == "negative").sum())


@st.cache_data
def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    file_map: Dict[str, str] = {
        "reviews": os.path.join(DATA_PATH, "cleaned_reviews.csv"),
        "profiles": os.path.join(DATA_PATH, "professor_profiles.csv"),
        "rec_db": os.path.join(DATA_PATH, "recommendation_db.csv"),
    }

    missing = [path for path in file_map.values() if not os.path.exists(path)]
    if missing:
        for path in missing:
            st.error(f"File not found: {path}")
        return None, None, None

    try:
        reviews = pd.read_csv(file_map["reviews"])
        profiles = pd.read_csv(file_map["profiles"])
        rec_db = pd.read_csv(file_map["rec_db"])
    except Exception as error:
        st.error(f"Error loading data: {error}")
        return None, None, None

    return reviews, profiles, rec_db


@st.cache_resource
def load_models():
    vec_path = os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(MODEL_PATH, "tfidf_search_matrix.pkl")
    missing = [path for path in [vec_path, matrix_path] if not os.path.exists(path)]
    if missing:
        for path in missing:
            st.error(f"Model file not found: {path}")
        return None, None

    try:
        vectorizer = joblib.load(vec_path)
        tfidf_matrix = joblib.load(matrix_path)
    except Exception as error:
        st.error(f"Error loading models: {error}")
        return None, None
    return vectorizer, tfidf_matrix


def ensure_bayesian_score(rec_df: pd.DataFrame) -> pd.DataFrame:
    result = rec_df.copy()
    if "bayesian_score" in result.columns and result["bayesian_score"].notna().any():
        return result

    if "avg_sentiment" not in result.columns:
        result["avg_sentiment"] = 0.5
    if "review_count" not in result.columns:
        result["review_count"] = 0

    result["avg_sentiment"] = pd.to_numeric(result["avg_sentiment"], errors="coerce").fillna(0.5)
    result["review_count"] = pd.to_numeric(result["review_count"], errors="coerce").fillna(0)

    c_value = float(result["avg_sentiment"].mean()) if not result.empty else 0.5
    m_value = float(result["review_count"].quantile(0.25)) if not result.empty else 1.0
    m_value = max(m_value, 1.0)

    denominator = result["review_count"] + m_value
    result["bayesian_score"] = (
        (result["review_count"] / denominator) * result["avg_sentiment"]
        + (m_value / denominator) * c_value
    )
    return result


def extract_query_snippet(text: object, query: str, window: int = 120) -> str:
    if not isinstance(text, str) or not text.strip():
        return "No comment snippet available."
    query_words = [word for word in query.split() if len(word) > 2]
    sentences = [chunk.strip() for chunk in text.replace("\n", ". ").split(".") if chunk.strip()]
    for sentence in sentences:
        if any(word in sentence for word in query_words):
            return sentence[:window] + ("..." if len(sentence) > window else "")
    return text[:window] + ("..." if len(text) > window else "")


def plot_radar_chart(professor_name: str, profiles_df: Optional[pd.DataFrame]) -> go.Figure:
    if profiles_df is None or profiles_df.empty:
        return go.Figure()

    professor_col = get_professor_column(profiles_df)
    if professor_col is None:
        return go.Figure()

    prof_rows = profiles_df[profiles_df[professor_col] == professor_name]
    if prof_rows.empty:
        return go.Figure()

    row = prof_rows.iloc[0]
    review_count = float(pd.to_numeric(row.get("review_count", 0), errors="coerce") or 0)
    max_reviews = float(pd.to_numeric(profiles_df.get("review_count", pd.Series([1])).max(), errors="coerce") or 1)
    max_reviews = max(max_reviews, 1.0)

    avg_sentiment = float(pd.to_numeric(row.get("avg_sentiment", 0), errors="coerce") or 0)
    if avg_sentiment > 1:
        avg_sentiment = avg_sentiment / 5

    numeric_score = pd.to_numeric(row.get("avg_numeric_score", avg_sentiment * 5), errors="coerce")
    numeric_score = float(numeric_score) if pd.notna(numeric_score) else 0.0

    values = [
        min(max(avg_sentiment * 5, 0), 5),
        min(max(numeric_score, 0), 5),
        min(max((review_count / max_reviews) * 5, 0), 5),
    ]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=["Avg Sentiment", "Numeric Score", "Popularity (Norm)"],
            fill="toself",
            name=professor_name,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        title=f"Profile: {professor_name}",
    )
    return fig


def plot_sentiment_dist(reviews_df: Optional[pd.DataFrame]):
    if reviews_df is None or reviews_df.empty:
        return px.pie(title="No Data Available")

    counts = sentiment_counts(reviews_df)
    if counts.empty or counts.sum() == 0:
        return px.pie(title="No Sentiment Data Available")

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Overall Sentiment Distribution",
        color=counts.index,
        color_discrete_map={
            "Positive": "#2a9d8f",
            "Negative": "#e63946",
            "Neutral": "#457b9d",
            "Unknown": "#9e9e9e",
        },
    )
    return fig
