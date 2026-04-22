import streamlit as st
import plotly.graph_objects as go
from utils import (
    get_cluster_column,
    get_professor_column,
    load_data,
    sentiment_pos_neg_counts,
)

st.title("⚖️ Professor Comparison")
st.markdown("Select two professors to compare their performance side-by-side.")

reviews, profiles, _ = load_data()

if profiles is not None and reviews is not None and not profiles.empty and not reviews.empty:
    profile_prof_col = get_professor_column(profiles)
    review_prof_col = get_professor_column(reviews)
    cluster_col = get_cluster_column(profiles)

    if profile_prof_col is None:
        st.error("No professor name column found in profile data.")
        st.stop()
    if review_prof_col is None:
        st.error("No professor name column found in review data.")
        st.stop()

    col_select1, col_select2 = st.columns(2)

    unique_profs = sorted(profiles[profile_prof_col].dropna().unique())
    if not unique_profs:
        st.error("No professor names found in profile data.")
        st.stop()

    with col_select1:
        prof1 = st.selectbox("Select First Professor", unique_profs, index=0)

    with col_select2:
        idx2 = 1 if len(unique_profs) > 1 else 0
        prof2 = st.selectbox("Select Second Professor", unique_profs, index=idx2)

    st.divider()

    col_info1, col_info2 = st.columns(2)
    
    for col, prof_name in zip([col_info1, col_info2], [prof1, prof2]):
        stats = profiles[profiles[profile_prof_col] == prof_name].iloc[0]
        with col:
            st.subheader(f"👤 {prof_name}")
            if cluster_col:
                st.write(f"**Cluster:** {stats[cluster_col]}")
            if "avg_numeric_score" in profiles.columns:
                st.metric("Avg Score", f"{stats['avg_numeric_score']:.2f} / 5")
            if "review_count" in profiles.columns:
                st.metric("Total Reviews", int(stats["review_count"]))

    st.divider()

    st.subheader("📊 Performance Comparison (Radar)")

    categories = ["Avg Sentiment", "Numeric Score", "Popularity"]

    fig = go.Figure()

    max_reviews = float(profiles["review_count"].max()) if "review_count" in profiles.columns else 1.0
    max_reviews = max(max_reviews, 1.0)

    for prof_name, color in zip([prof1, prof2], ["#636EFA", "#EF553B"]):
        prof_data = profiles[profiles[profile_prof_col] == prof_name].iloc[0]
        review_count = float(prof_data.get("review_count", 0))
        avg_sentiment = float(prof_data.get("avg_sentiment", 0))
        if avg_sentiment > 1:
            avg_sentiment = avg_sentiment / 5
        avg_score = float(prof_data.get("avg_numeric_score", avg_sentiment * 5))
        popularity = (review_count / max_reviews) * 5

        values = [
            max(min(avg_sentiment * 5, 5), 0),
            max(min(avg_score, 5), 0),
            popularity
        ]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=prof_name,
            line_color=color,
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💬 Quick Sentiment Overview")
    col_sent1, col_sent2 = st.columns(2)

    for col, prof_name in zip([col_sent1, col_sent2], [prof1, prof2]):
        prof_reviews = reviews[reviews[review_prof_col] == prof_name]
        pos_count, neg_count = sentiment_pos_neg_counts(prof_reviews)

        with col:
            st.write(f"Positive Reviews: ✅ {pos_count}")
            st.write(f"Negative Reviews: ❌ {neg_count}")
else:
    st.error("Failed to load data. Please check utils.py")
