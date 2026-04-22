# elmous-yaar-data-mining

Persian professor-review mining project with notebook workflows, reusable Python pipeline modules, and a Streamlit dashboard.

## Project Layout

- `data/raw`: Telegram export input data
- `data/processed`: parsed, cleaned, and feature-engineered datasets
- `models`: saved vectorizers and matrices used by the dashboard
- `src`: reusable pipeline modules (`parsing`, `cleaning`, `features`, `sentiment`)
- `app`: Streamlit dashboard with overview, search, comparison, and recommender pages
- `notebooks`: exploratory and model-development notebooks

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Pipeline Modules

```python
import pandas as pd
from src.cleaning import clean_reviews_dataframe
from src.features import build_professor_profiles, build_recommendation_db

reviews = pd.read_csv("data/processed/cleaned_reviews.csv")
cleaned = clean_reviews_dataframe(reviews)
profiles = build_professor_profiles(cleaned)
rec_db = build_recommendation_db(cleaned, profiles)
```
