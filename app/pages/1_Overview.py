import streamlit as st
import plotly.express as px
from utils import (
    get_cluster_column,
    get_professor_column,
    load_data,
    plot_sentiment_dist,
)

st.title("📊 Overview & Statistics")

reviews, profiles, _ = load_data()

if reviews is not None and profiles is not None and not reviews.empty and not profiles.empty:
    professor_col = get_professor_column(profiles)
    cluster_col = get_cluster_column(profiles)
    tracked_professors = profiles[professor_col].nunique() if professor_col else len(profiles)
    avg_score = profiles["avg_numeric_score"].mean() if "avg_numeric_score" in profiles.columns else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(reviews))
    col2.metric("Professors Tracked", tracked_professors)
    col3.metric(
        "Avg University Score",
        f"{avg_score:.2f}/5" if avg_score is not None else "N/A",
    )

    st.divider()

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Sentiment Distribution")
        st.plotly_chart(plot_sentiment_dist(reviews), use_container_width=True)

    with col_chart2:
        st.subheader("Numeric Score Distribution")
        if "avg_numeric_score" in profiles.columns:
            fig = px.histogram(
                profiles,
                x="avg_numeric_score",
                nbins=20,
                title="Distribution of Average Scores",
                color_discrete_sequence=["#636EFA"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("`avg_numeric_score` is not available in profile data.")

    st.subheader("🏆 Top Rated Professors")
    if "avg_numeric_score" in profiles.columns:
        top_profs = profiles.sort_values(by="avg_numeric_score", ascending=False).head(10).copy()
        table_columns = []
        if professor_col:
            table_columns.append(professor_col)
        table_columns.extend([col for col in ["avg_numeric_score", "review_count"] if col in top_profs.columns])
        if cluster_col:
            table_columns.append(cluster_col)
        st.dataframe(top_profs[table_columns], hide_index=True, use_container_width=True)
    else:
        st.info("Top professors cannot be ranked because `avg_numeric_score` is missing.")
else:
    st.error("Failed to load overview data.")
