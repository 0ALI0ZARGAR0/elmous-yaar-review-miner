from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SENTIMENT_LABEL_TO_SCORE = {
    "recommended": 1.0,
    "positive": 1.0,
    "neutral": 0.5,
    "no_idea": 0.4,
    "not_recommended": 0.0,
    "negative": 0.0,
}


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _sentiment_series(df: pd.DataFrame) -> pd.Series:
    if "sentiment_score" in df.columns:
        score = pd.to_numeric(df["sentiment_score"], errors="coerce")
        if score.notna().any():
            return score.fillna(score.median() if score.notna().any() else 0.5).clip(0, 1)
    if "sentiment_label" in df.columns:
        labels = df["sentiment_label"].astype(str).str.strip().str.lower()
        return labels.map(SENTIMENT_LABEL_TO_SCORE).fillna(0.4)
    return pd.Series(np.repeat(0.5, len(df)), index=df.index, dtype=float)


def _numeric_score_series(df: pd.DataFrame) -> pd.Series:
    if "rating_mean" in df.columns:
        rating = pd.to_numeric(df["rating_mean"], errors="coerce")
    else:
        rating_columns = [column for column in ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5", "rating_6"] if column in df.columns]
        if rating_columns:
            rating = df[rating_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        else:
            rating = pd.Series(np.repeat(np.nan, len(df)), index=df.index)
    if rating.notna().any():
        max_rating = float(rating.max())
        if max_rating > 5:
            rating = rating / 2
        return rating.clip(0, 5)
    sentiment_based = _sentiment_series(df) * 5
    return sentiment_based.clip(0, 5)


def build_professor_profiles(reviews_df: pd.DataFrame, min_reviews: int = 1) -> pd.DataFrame:
    if reviews_df.empty:
        return pd.DataFrame(
            columns=["professor_name_raw", "avg_sentiment", "review_count", "avg_numeric_score", "log_review_count"]
        )

    professor_col = _first_existing_column(reviews_df, ["professor_name_raw", "teacher_name", "professor_name"])
    if professor_col is None:
        raise ValueError("No professor name column found in reviews dataframe.")

    work_df = reviews_df.copy()
    work_df = work_df[work_df[professor_col].notna()]
    work_df["sentiment_value"] = _sentiment_series(work_df)
    work_df["numeric_score_value"] = _numeric_score_series(work_df)

    profiles = (
        work_df.groupby(professor_col, as_index=False)
        .agg(
            avg_sentiment=("sentiment_value", "mean"),
            review_count=(professor_col, "size"),
            avg_numeric_score=("numeric_score_value", "mean"),
        )
        .sort_values("review_count", ascending=False)
    )
    profiles = profiles[profiles["review_count"] >= int(min_reviews)].copy()
    profiles["log_review_count"] = np.log1p(profiles["review_count"])

    if professor_col != "professor_name_raw":
        profiles = profiles.rename(columns={professor_col: "professor_name_raw"})

    return profiles.reset_index(drop=True)


def calculate_bayesian_score(
    df: pd.DataFrame,
    rating_col: str = "avg_sentiment",
    votes_col: str = "review_count",
    quantile: float = 0.25,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    work_df = df.copy()
    work_df[rating_col] = pd.to_numeric(work_df[rating_col], errors="coerce").fillna(0.5).clip(0, 1)
    work_df[votes_col] = pd.to_numeric(work_df[votes_col], errors="coerce").fillna(0).clip(lower=0)

    c_value = float(work_df[rating_col].mean())
    m_value = float(work_df[votes_col].quantile(quantile))
    m_value = max(m_value, 1.0)

    denominator = work_df[votes_col] + m_value
    return ((work_df[votes_col] / denominator) * work_df[rating_col]) + ((m_value / denominator) * c_value)


def build_recommendation_db(
    reviews_df: pd.DataFrame,
    profiles_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if profiles_df is None:
        profiles_df = build_professor_profiles(reviews_df)
    if profiles_df.empty:
        return pd.DataFrame()

    review_professor_col = _first_existing_column(reviews_df, ["professor_name_raw", "teacher_name", "professor_name"])
    profile_professor_col = _first_existing_column(profiles_df, ["professor_name_raw", "teacher_name", "professor_name"])
    comment_col = _first_existing_column(reviews_df, ["comment_text", "review_text", "text"])

    if review_professor_col is None or profile_professor_col is None:
        raise ValueError("Professor name columns are required in both review and profile dataframes.")

    if comment_col is None:
        comment_agg = pd.DataFrame({"professor_name_raw": profiles_df[profile_professor_col], "comment_text": ""})
    else:
        comment_agg = (
            reviews_df[[review_professor_col, comment_col]]
            .dropna(subset=[review_professor_col])
            .astype({review_professor_col: str})
            .groupby(review_professor_col)[comment_col]
            .agg(lambda series: " ".join(series.fillna("").astype(str)).strip())
            .reset_index(name="comment_text")
            .rename(columns={review_professor_col: "professor_name_raw"})
        )

    profiles_work = profiles_df.copy()
    if profile_professor_col != "professor_name_raw":
        profiles_work = profiles_work.rename(columns={profile_professor_col: "professor_name_raw"})

    rec_db = profiles_work.merge(comment_agg, how="left", on="professor_name_raw")
    rec_db["comment_text"] = rec_db["comment_text"].fillna("")
    rec_db["bayesian_score"] = calculate_bayesian_score(rec_db)
    return rec_db.reset_index(drop=True)


def build_tfidf_index(
    rec_db: pd.DataFrame,
    text_col: str = "comment_text",
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
):
    if text_col not in rec_db.columns:
        raise ValueError(f"Column `{text_col}` not found in recommendation dataframe.")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)
    matrix = vectorizer.fit_transform(rec_db[text_col].fillna("").astype(str))
    return vectorizer, matrix
