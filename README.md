# Sentiment-Aware Movie Recommender (FDS Project)

This project is a Foundations of Data Science (FDS) capstone that blends a content-based movie recommendation engine with a lightweight sentiment analyzer. It offers both a terminal interface for quick experimentation and a Flask web client for interactive exploration.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Flask](https://img.shields.io/badge/Flask-2.3-red) ![Pandas](https://img.shields.io/badge/Pandas-2.x-lightgrey) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5+-orange)

---

## Features
- Content-based movie recommendations powered by a `CountVectorizer` + cosine similarity pipeline.
- Sentiment classifier (Multinomial Naive Bayes) for free-form movie reviews.
- Autocomplete/search endpoints to support the web UI.
- Clean separation between raw data, processed artifacts, trained models, and application code.
- Ready-to-run CLI (`python cli.py`) and web server (`python app.py`).

---

## Project Structure
```
.
├── cli.py                     # CLI entry point
├── app.py                     # Flask web app
├── requirements.txt
├── data/
│   ├── processed/
│   │   └── main_data.csv      # Feature matrix exported for inference
│   └── raw/                   # Source datasets (ignored by git)
├── models/
│   ├── nlp_model.pkl          # Sentiment classifier
│   └── tranform.pkl           # Vectorizer for sentiment model
├── src/
│   ├── __init__.py
│   └── movie_recommender.py   # Recommender core + terminal UI
├── static/                    # Front-end assets
└── templates/                 # Jinja templates for Flask UI
```

---

## Getting Started
```bash
git clone https://github.com/<your-handle>/SentimentAwareMovieRecommender.git
cd SentimentAwareMovieRecommender
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables
No secrets are required. If you plan to integrate external APIs (TMDB, etc.), store keys inside a `.env` file and load them in `app.py`.

---

## Running the Apps

### 1. Terminal (CLI)
```bash
source venv/bin/activate
python cli.py
```
Use the interactive menu to:
- Search movies
- Request top-N recommendations
- Run sentiment analysis on a review

### 2. Flask Web UI
```bash
source venv/bin/activate
python app.py
```
Open `http://localhost:5002` (or the port shown in the console). The web client exposes search, recommendations, sentiment analysis, and movie detail panels backed by the same recommender core.

---

## Data & Models
- `data/raw/`: Contains the original IMDB + TMDB exports used during experimentation. The folder is ignored by git to keep the repo lightweight.
- `data/processed/main_data.csv`: Final feature set used by the recommender at inference time (tracked).
- `models/`: Serialized scikit-learn model and vectorizer required for sentiment analysis (tracked).

If you would like to regenerate the processed dataset or retrain the classifier, drop new files into `data/raw/` and follow the notebook or scripts that produced `main_data.csv` (not included yet—feel free to add your own experimentation notebooks under `notebooks/`).

---

## Testing & Quality
- Dataset loading, cosine similarity construction, and sentiment predictions are executed automatically every time the CLI or Flask app starts. If any assets are missing you will see a descriptive error message.
- For linting/formatting, run `ruff` or `flake8` over the `src/` directory (config not included; add your preferred tooling).

---

---

## License
MIT – feel free to reuse and adapt for your own FDS submission. Please credit the original authors when publishing new work derived from this repository.
