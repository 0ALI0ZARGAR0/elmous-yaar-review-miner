import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Professor Analytics Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University Professor Analytics & Recommender")

st.markdown("""
### Welcome to the Smart Professor Assistant
This dashboard provides deep insights into student reviews using **Natural Language Processing (NLP)** and **Machine Learning**.

#### 👈 Select a page from the sidebar to explore:
* **Overview:** General statistics and university-wide trends.
* **Search & Filter:** Find detailed profiles for specific professors.
* **Compare:** Head-to-head comparison of professors.
* **Smart Recommender:** Ask for a professor using natural language (e.g., "Easy exams").
""")

st.info("Data processed from thousands of student reviews.")
