# Sentiment-Aware Movie Recommender (FDS Project)

This project is a Foundations of Data Science (FDS) capstone that blends a content-based movie recommendation engine with a lightweight sentiment analyzer. It offers both a terminal interface for quick experimentation and a Flask web client for interactive exploration.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Flask](https://img.shields.io/badge/Flask-2.3-red) ![Pandas](https://img.shields.io/badge/Pandas-2.x-lightgrey) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5+-orange)

---

## Features
- Content-based movie recommendations powered by a `CountVectorizer` + cosine similarity pipeline.
- Sentiment classifier (Multinomial Naive Bayes) for free-form movie reviews.
- **Intelligent autocomplete** with fuzzy matching across all interfaces (GUI, Web, CLI).
- Autocomplete/search endpoints to support the web UI.
- Clean separation between raw data, processed artifacts, trained models, and application code.
- Ready-to-run CLI (`python cli.py`) and web server (`python app.py`).
- Modern desktop GUI with real-time suggestions (`python gui.py`).

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
- Search movies with **Tab completion** (press Tab to autocomplete movie names)
- Request top-N recommendations with intelligent suggestions
- Run sentiment analysis on a review
- **Tip**: Start typing a movie name and press Tab to see all matches!

### 2. Flask Web UI
```bash
source venv/bin/activate
python app.py
```
Open `http://localhost:5002` (or the port shown in the console). The web client exposes:
- **Live autocomplete** in all search fields (start typing to see suggestions)
- Movie recommendations with visual similarity scores
- Sentiment analysis with confidence metrics
- Movie detail panels backed by the same recommender core

### 3. Modern Desktop GUI
```bash
source venv/bin/activate
python gui.py
```
Launch a beautiful, modern desktop application with:
- **Real-time autocomplete** - Fuzzy-matched movie suggestions as you type
- **Interactive expandable cards** - Click to expand/collapse recommendation details
- **Dark theme with transparency** - Sleek, contemporary design
- **Three-tab interface**:
  - 🔍 Search & Recommend - Get personalized movie recommendations with expandable cards
  - 💭 Sentiment Analysis - Analyze movie reviews with AI
  - ℹ️ Movie Info - View detailed movie information
- **Visual feedback** - Color-coded sentiment results with confidence meters
- **Action buttons** - Quick access to full details and similar movies from each card

---

## Data & Models
- `data/raw/`: Contains the original IMDB + TMDB exports used during experimentation. The folder is ignored by git to keep the repo lightweight.
- `data/processed/main_data.csv`: Final feature set used by the recommender at inference time (tracked).
- `models/`: Serialized scikit-learn model and vectorizer required for sentiment analysis (tracked).

If you would like to regenerate the processed dataset or retrain the classifier, drop new files into `data/raw/` and follow the notebook or scripts that produced `main_data.csv` (not included yet—feel free to add your own experimentation notebooks under `notebooks/`).

---

## Autocomplete & Search Features

All three interfaces support intelligent autocomplete with fuzzy matching! See **[AUTOCOMPLETE_FEATURES.md](AUTOCOMPLETE_FEATURES.md)** for detailed documentation on:
- How autocomplete works in each interface
- Keyboard shortcuts and tips
- Fuzzy matching algorithm
- Performance characteristics
- Usage examples

---

## Testing & Quality
- Dataset loading, cosine similarity construction, and sentiment predictions are executed automatically every time the CLI or Flask app starts. If any assets are missing you will see a descriptive error message.
- For linting/formatting, run `ruff` or `flake8` over the `src/` directory (config not included; add your preferred tooling).

---

---

## License
MIT – feel free to reuse and adapt for your own FDS submission. Please credit the original authors when publishing new work derived from this repository.
