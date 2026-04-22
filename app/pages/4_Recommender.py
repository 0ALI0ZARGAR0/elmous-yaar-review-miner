import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    ensure_bayesian_score,
    extract_query_snippet,
    get_cluster_column,
    get_comment_column,
    get_professor_column,
    load_data,
    load_models,
)

st.title("🤖 Smart Professor Recommender")
st.markdown("Describe what you are looking for (e.g., *'خوش اخلاق و نمره خوب'*).")

_, _, rec_db = load_data()
vectorizer, tfidf_matrix = load_models()

if rec_db is not None and vectorizer is not None and tfidf_matrix is not None and not rec_db.empty:
    rec_df = ensure_bayesian_score(rec_db)
    professor_col = get_professor_column(rec_df) or "professor_name_raw"
    cluster_col = get_cluster_column(rec_df)
    comment_col = get_comment_column(rec_df)

    query = st.text_input("Enter your preferences:", "")
    top_n = st.slider("Number of matches", min_value=3, max_value=10, value=5, step=1)
    semantic_weight = st.slider("Semantic match weight", min_value=0.1, max_value=0.9, value=0.7, step=0.1)
    quality_weight = round(1.0 - semantic_weight, 1)
    st.caption(f"Quality weight is set to `{quality_weight}`.")

    if st.button("Find Professors"):
        query = query.strip()
        if not query:
            st.warning("Please enter a query.")
            st.stop()

        try:
            query_vec = vectorizer.transform([query])
            similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

            if len(similarity) != len(rec_df):
                aligned_length = min(len(similarity), len(rec_df))
                similarity = similarity[:aligned_length]
                rec_df = rec_df.iloc[:aligned_length].copy()
                st.warning("Recommendation database and TF-IDF matrix were misaligned; results were auto-aligned.")

            results = rec_df.copy()
            results["semantic_score"] = similarity
            results["final_score"] = (
                results["semantic_score"] * semantic_weight
                + results["bayesian_score"] * quality_weight
            )
            results = (
                results[results["semantic_score"] > 0]
                .sort_values(by="final_score", ascending=False)
                .head(top_n)
            )

            if results.empty:
                st.warning("No matches found. Try different keywords.")
            else:
                for _, row in results.iterrows():
                    prof_name = row.get(professor_col, "Unknown Professor")
                    snippet = extract_query_snippet(row.get(comment_col, "") if comment_col else "", query)
                    with st.expander(f"🏆 {prof_name} (Match: {row['final_score']:.2f})"):
                        if cluster_col:
                            st.write(f"**Profile:** {row.get(cluster_col, 'N/A')}")
                        if "avg_numeric_score" in results.columns:
                            st.write(f"**Average Score:** {row['avg_numeric_score']:.2f}")
                        st.write(f"**Why this match?** {snippet}")
                        st.progress(float(max(min(row["semantic_score"], 1), 0)), text="Semantic relevance")
        except Exception as error:
            st.error(f"Error processing query: {error}")
else:
    st.error("System is not ready. Data or Models missing.")
