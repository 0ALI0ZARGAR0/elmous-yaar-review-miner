import streamlit as st
from utils import (
    get_cluster_column,
    get_comment_column,
    get_professor_column,
    load_data,
    plot_radar_chart,
)
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("🔍 Search & Professor Profile")

reviews, profiles, _ = load_data()

if profiles is not None and reviews is not None and not profiles.empty and not reviews.empty:
    st.sidebar.header("Filters")

    profile_prof_col = get_professor_column(profiles)
    review_prof_col = get_professor_column(reviews)
    cluster_col = get_cluster_column(profiles)
    comment_col = get_comment_column(reviews)

    if profile_prof_col is None:
        st.error("No professor name column found in profile data.")
        st.stop()
    if review_prof_col is None:
        st.error("No professor name column found in reviews data.")
        st.stop()

    unique_profs = sorted(profiles[profile_prof_col].dropna().unique())
    if not unique_profs:
        st.error("No professor names found in profile data.")
        st.stop()
    selected_prof = st.sidebar.selectbox("Select Professor", unique_profs, index=0)

    if selected_prof:
        prof_stats = profiles[profiles[profile_prof_col] == selected_prof].iloc[0]
        prof_reviews = reviews[reviews[review_prof_col] == selected_prof]

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
        with col2:
            st.header(selected_prof)
            if cluster_col:
                st.markdown(f"**Cluster:** {prof_stats[cluster_col]}")
            if "avg_numeric_score" in profiles.columns:
                st.markdown(f"**Average Score:** ⭐ {prof_stats['avg_numeric_score']:.2f} / 5")
            if "review_count" in profiles.columns:
                st.markdown(f"**Total Reviews:** {int(prof_stats['review_count'])}")

        st.divider()

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.plotly_chart(plot_radar_chart(selected_prof, profiles), use_container_width=True)

        with col_c2:
            st.subheader("☁️ What students say (WordCloud)")

            if not prof_reviews.empty and comment_col:
                text = " ".join(prof_reviews[comment_col].dropna().astype(str))
                try:
                    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as error:
                    st.error(f"Could not generate word cloud: {error}")
            else:
                st.info("No textual reviews available for word cloud.")

        st.subheader("📝 Recent Student Reviews")
        if not prof_reviews.empty and comment_col:
            for txt in prof_reviews[comment_col].dropna().head(5):
                st.info(txt)
        else:
            st.write("No reviews found.")

else:
    st.error("Failed to load data. Please check processed files in `data/processed`.")
