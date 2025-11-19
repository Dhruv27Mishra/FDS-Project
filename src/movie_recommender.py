#!/usr/bin/env python3
"""
Sentiment-Aware Movie Recommendation System
A content-based movie recommender system integrated with sentiment analysis
to improve user experience in movie selection.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import sys
import readline
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

# Transformer-based imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Falling back to CountVectorizer.")

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Falling back to Naive Bayes.")


# ANSI Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'      # Magenta
    BLUE = '\033[94m'        # Blue
    CYAN = '\033[96m'        # Cyan
    GREEN = '\033[92m'       # Green
    YELLOW = '\033[93m'      # Yellow
    RED = '\033[91m'         # Red
    BOLD = '\033[1m'         # Bold
    UNDERLINE = '\033[4m'    # Underline
    END = '\033[0m'          # Reset
    
    # Custom colors
    SUCCESS = '\033[92m'     # Green
    ERROR = '\033[91m'       # Red
    WARNING = '\033[93m'     # Yellow
    INFO = '\033[96m'        # Cyan
    TITLE = '\033[1m\033[95m'  # Bold Magenta
    HIGHLIGHT = '\033[1m\033[94m'  # Bold Blue
    SCORE = '\033[92m'       # Green for scores


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DATA_PATH = DATA_DIR / "main_data.csv"
SENTIMENT_MODEL_PATH = MODELS_DIR / "nlp_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tranform.pkl"


class MovieRecommender:
    """Main class for the Movie Recommendation System with Sentiment Analysis"""
    
    def __init__(self, use_transformers=True):
        """
        Initialize the recommender system
        
        Args:
            use_transformers: If True, use transformer models (Sentence Transformers + DistilBERT)
                             If False, use legacy models (CountVectorizer + Naive Bayes)
        """
        self.data = None
        self.similarity = None
        self.embeddings = None
        self.clf = None
        self.vectorizer = None
        self.sentiment_pipeline = None
        self.sentence_model = None
        self.use_transformers = use_transformers and SENTENCE_TRANSFORMERS_AVAILABLE and TRANSFORMERS_AVAILABLE
        self._load_data()
        self._load_models()
    
    def _load_data(self):
        """Load movie data and create similarity matrix or embeddings"""
        print("Loading movie data...")
        try:
            self.data = pd.read_csv(PROCESSED_DATA_PATH)
            print(f"✓ Loaded {len(self.data)} movies")
            
            if self.use_transformers:
                # Use Sentence Transformers for semantic embeddings
                print("Creating semantic embeddings with Sentence Transformers...")
                try:
                    # Use a fast, efficient model
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                    print("✓ Sentence Transformer model loaded")
                    
                    # Create descriptive text for each movie
                    descriptions = []
                    for _, row in self.data.iterrows():
                        # Combine all features into a descriptive text
                        desc_parts = []
                        if pd.notna(row.get('genres')):
                            desc_parts.append(str(row['genres']))
                        if pd.notna(row.get('director_name')):
                            desc_parts.append(f"directed by {row['director_name']}")
                        if pd.notna(row.get('actor_1_name')):
                            desc_parts.append(f"starring {row['actor_1_name']}")
                        if pd.notna(row.get('actor_2_name')):
                            desc_parts.append(f"and {row['actor_2_name']}")
                        if pd.notna(row.get('actor_3_name')):
                            desc_parts.append(f"and {row['actor_3_name']}")
                        
                        desc = " ".join(desc_parts) if desc_parts else str(row.get('comb', ''))
                        descriptions.append(desc)
                    
                    # Generate embeddings
                    self.embeddings = self.sentence_model.encode(descriptions, show_progress_bar=True)
                    print(f"✓ Created {len(self.embeddings)} semantic embeddings")
                    
                    # Pre-compute similarity matrix for faster recommendations
                    print("Computing similarity matrix from embeddings...")
                    self.similarity = cosine_similarity(self.embeddings)
                    print("✓ Similarity matrix computed")
                    
                except Exception as e:
                    print(f"Warning: Failed to load Sentence Transformers: {e}")
                    print("Falling back to CountVectorizer...")
                    self.use_transformers = False
                    self._load_data_legacy()
            else:
                self._load_data_legacy()
                
        except FileNotFoundError:
            print(f"Error: {PROCESSED_DATA_PATH} not found!")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def _load_data_legacy(self):
        """Legacy method using CountVectorizer"""
        print("Computing similarity matrix with CountVectorizer...")
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(self.data['comb'])
        self.similarity = cosine_similarity(count_matrix)
        print("✓ Similarity matrix computed")
    
    def _load_models(self):
        """Load pre-trained sentiment analysis models"""
        print("Loading sentiment analysis models...")
        
        if self.use_transformers:
            # Use DistilBERT for sentiment analysis
            try:
                # Use GPU if available, else CPU
                device = 0 if torch.cuda.is_available() else -1
                
                # Use a fast, accurate sentiment model
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device
                )
                print("✓ DistilBERT sentiment model loaded")
                if device >= 0:
                    print("  (Using GPU acceleration)")
                else:
                    print("  (Using CPU)")
            except Exception as e:
                print(f"Warning: Failed to load DistilBERT: {e}")
                print("Falling back to legacy Naive Bayes model...")
                self.use_transformers = False
                self._load_models_legacy()
        else:
            self._load_models_legacy()
    
    def _load_models_legacy(self):
        """Legacy method using Naive Bayes"""
        try:
            with open(SENTIMENT_MODEL_PATH, 'rb') as model_file:
                self.clf = pickle.load(model_file)
            with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
                self.vectorizer = pickle.load(vectorizer_file)
            print("✓ Sentiment analysis models loaded (Naive Bayes)")
        except FileNotFoundError:
            print("Warning: Sentiment analysis models not found. Sentiment analysis will be disabled.")
            self.clf = None
            self.vectorizer = None
        except Exception as e:
            print(f"Warning: Error loading models: {e}. Sentiment analysis will be disabled.")
            self.clf = None
            self.vectorizer = None
    
    def get_movie_suggestions(self, query: str, limit: int = 10) -> List[str]:
        """Get movie suggestions based on partial query"""
        if not query:
            return []
        
        query_lower = query.lower()
        matches = self.data[
            self.data['movie_title'].str.lower().str.contains(query_lower, na=False)
        ]['movie_title'].str.capitalize().tolist()
        
        return matches[:limit]
    
    def recommend_movies(self, movie_title: str, num_recommendations: int = 10) -> Tuple[List[str], Optional[str]]:
        """
        Get movie recommendations based on a movie title
        
        Args:
            movie_title: Title of the movie to get recommendations for
            num_recommendations: Number of recommendations to return
            
        Returns:
            Tuple of (list of recommended movies, error message if any)
        """
        movie_title_lower = movie_title.lower()
        
        # Check if movie exists in database
        if movie_title_lower not in self.data['movie_title'].str.lower().values:
            return [], f"Movie '{movie_title}' not found in database. Please check spelling or try another movie."
        
        # Find the index of the movie
        movie_idx = self.data[self.data['movie_title'].str.lower() == movie_title_lower].index[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity[movie_idx]))
        
        # Sort by similarity (descending)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations (excluding the movie itself)
        top_movies = similarity_scores[1:num_recommendations + 1]
        
        # Extract movie titles
        recommendations = []
        for idx, score in top_movies:
            recommendations.append({
                'title': self.data.iloc[idx]['movie_title'],
                'similarity': score,
                'director': self.data.iloc[idx]['director_name'],
                'genres': self.data.iloc[idx]['genres'],
                'actors': [
                    self.data.iloc[idx]['actor_1_name'],
                    self.data.iloc[idx]['actor_2_name'],
                    self.data.iloc[idx]['actor_3_name']
                ]
            })
        
        return recommendations, None
    
    def analyze_sentiment(self, review: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a movie review
        
        Args:
            review: Text review to analyze
            
        Returns:
            Tuple of (sentiment label, confidence score)
        """
        if self.use_transformers:
            # Use DistilBERT for sentiment analysis
            if self.sentiment_pipeline is None:
                return "Sentiment analysis unavailable", 0.0
            
            try:
                # Truncate review if too long (BERT models have token limits)
                max_length = 512
                if len(review) > max_length:
                    review = review[:max_length]
                
                # Get sentiment prediction
                result = self.sentiment_pipeline(review)[0]
                
                # Extract label and score
                label = result['label']
                score = result['score']
                
                # Map to standard format
                if 'POSITIVE' in label.upper() or 'POS' in label.upper():
                    sentiment = "Positive"
                else:
                    sentiment = "Negative"
                
                return sentiment, score
            except Exception as e:
                return f"Error: {e}", 0.0
        else:
            # Legacy Naive Bayes method
            if self.clf is None or self.vectorizer is None:
                return "Sentiment analysis unavailable", 0.0
            
            try:
                # Transform review
                review_vector = self.vectorizer.transform([review])
                
                # Predict sentiment
                prediction = self.clf.predict(review_vector)[0]
                probabilities = self.clf.predict_proba(review_vector)[0]
                
                # Get confidence
                confidence = max(probabilities)
                
                # Map prediction to label
                sentiment = "Positive" if prediction == 1 else "Negative"
                
                return sentiment, confidence
            except Exception as e:
                return f"Error: {e}", 0.0
    
    def get_movie_info(self, movie_title: str) -> Optional[dict]:
        """Get detailed information about a movie"""
        movie_title_lower = movie_title.lower()
        
        if movie_title_lower not in self.data['movie_title'].str.lower().values:
            return None
        
        movie_row = self.data[self.data['movie_title'].str.lower() == movie_title_lower].iloc[0]
        
        return {
            'title': movie_row['movie_title'],
            'director': movie_row['director_name'],
            'genres': movie_row['genres'],
            'actors': [
                movie_row['actor_1_name'],
                movie_row['actor_2_name'],
                movie_row['actor_3_name']
            ]
        }


class TerminalInterface:
    """Interactive terminal interface for the movie recommender"""
    
    def __init__(self, recommender: MovieRecommender):
        self.recommender = recommender
        self.setup_autocomplete()
    
    def setup_autocomplete(self):
        """Setup readline autocomplete for movie titles"""
        # Get all movie titles for autocomplete
        self.all_movies = self.recommender.data['movie_title'].str.capitalize().tolist()
        
        # Configure readline
        readline.set_completer(self.movie_completer)
        readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(' \t\n')
    
    def movie_completer(self, text, state):
        """Autocomplete function for movie titles"""
        if not text:
            matches = self.all_movies[:]
        else:
            # Case-insensitive fuzzy matching
            text_lower = text.lower()
            matches = [
                movie for movie in self.all_movies 
                if text_lower in movie.lower()
            ]
        
        # Return the state-th match
        if state < len(matches):
            return matches[state]
        else:
            return None
    
    def get_input_with_suggestions(self, prompt: str, show_suggestions: bool = True) -> str:
        """Get input with live suggestions"""
        print(prompt, end='', flush=True)
        
        if show_suggestions:
            # Enable readline completion
            readline.set_completer(self.movie_completer)
        else:
            # Disable completion
            readline.set_completer(None)
        
        try:
            user_input = input().strip()
            
            # If input is partial, show suggestions
            if show_suggestions and user_input and len(user_input) >= 2:
                suggestions = self.recommender.get_movie_suggestions(user_input, limit=5)
                if suggestions and user_input.lower() not in [s.lower() for s in suggestions]:
                    print(f"\n{Colors.INFO}💡 Did you mean one of these? (Press Tab to autocomplete){Colors.END}")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"   {Colors.YELLOW}{i}.{Colors.END} {suggestion}")
                    
                    # Ask if user wants to select from suggestions
                    selection = input(f"\n{Colors.CYAN}Select a number (1-5) or press Enter to continue:{Colors.END} ").strip()
                    if selection.isdigit() and 1 <= int(selection) <= len(suggestions):
                        user_input = suggestions[int(selection) - 1]
                        print(f"{Colors.SUCCESS}✓ Selected: {user_input}{Colors.END}")
            
            return user_input
        except EOFError:
            return ""
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print application header"""
        print(f"{Colors.TITLE}{'=' * 70}{Colors.END}")
        print(f"{Colors.TITLE}{' ' * 15}MOVIE RECOMMENDATION SYSTEM{Colors.END}")
        print(f"{Colors.TITLE}{' ' * 10}with Sentiment Analysis{Colors.END}")
        print(f"{Colors.TITLE}{'=' * 70}{Colors.END}")
        print()
    
    def print_menu(self):
        """Print main menu options"""
        print(f"\n{Colors.CYAN}{'-' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}MAIN MENU{Colors.END}")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
        print(f"{Colors.YELLOW}1.{Colors.END} Get Movie Recommendations")
        print(f"{Colors.YELLOW}2.{Colors.END} Analyze Review Sentiment")
        print(f"{Colors.YELLOW}3.{Colors.END} Search Movies")
        print(f"{Colors.YELLOW}4.{Colors.END} View Movie Details")
        print(f"{Colors.YELLOW}5.{Colors.END} View Model Evaluation Metrics")
        print(f"{Colors.YELLOW}6.{Colors.END} Exit")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
    
    def get_movie_recommendations(self):
        """Interactive flow for getting movie recommendations"""
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}MOVIE RECOMMENDATIONS{Colors.END}")
        print(f"{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"\n{Colors.INFO}💡 Tip: Start typing and press Tab to autocomplete movie names{Colors.END}")
        
        movie_title = self.get_input_with_suggestions(f"\n{Colors.CYAN}Enter a movie title:{Colors.END} ", show_suggestions=True)
        
        if not movie_title:
            print(f"{Colors.ERROR}Error: Movie title cannot be empty!{Colors.END}")
            return
        
        print(f"\n{Colors.INFO}Searching for recommendations based on '{Colors.BOLD}{movie_title}{Colors.END}{Colors.INFO}'...{Colors.END}")
        recommendations, error = self.recommender.recommend_movies(movie_title)
        
        if error:
            print(f"\n{Colors.ERROR}❌ {error}{Colors.END}")
            
            # Suggest similar movies
            suggestions = self.recommender.get_movie_suggestions(movie_title, limit=5)
            if suggestions:
                print(f"\n{Colors.WARNING}💡 Did you mean one of these?{Colors.END}")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"   {Colors.YELLOW}{i}.{Colors.END} {suggestion}")
            return
        
        print(f"\n{Colors.SUCCESS}✓ Found {len(recommendations)} recommendations:{Colors.END}\n")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
        print(f"{Colors.INFO}💡 Tip: Enter a number (1-{len(recommendations)}) to see full details, or press Enter to continue{Colors.END}")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
        
        # Display recommendations in compact format
        for i, movie in enumerate(recommendations, 1):
            print(f"\n{Colors.HIGHLIGHT}{i}. {movie['title']}{Colors.END}")
            print(f"   {Colors.SCORE}Match Score: {movie['similarity']:.2%}{Colors.END}")
        
        print(f"\n{Colors.CYAN}{'-' * 70}{Colors.END}")
        
        # Ask if user wants to see details
        choice = input(f"\n{Colors.YELLOW}Enter number to expand (or press Enter to skip):{Colors.END} ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(recommendations):
                self.show_expanded_movie(recommendations[idx], idx + 1)
            else:
                print(f"{Colors.ERROR}Invalid number!{Colors.END}")
    
    def show_expanded_movie(self, movie, index):
        """Show detailed information for a specific movie"""
        print(f"\n{Colors.BLUE}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}DETAILED VIEW - #{index}{Colors.END}")
        print(f"{Colors.BLUE}{'=' * 70}{Colors.END}")
        
        print(f"\n{Colors.BOLD}Title:{Colors.END} {Colors.HIGHLIGHT}{movie['title']}{Colors.END}")
        print(f"{Colors.BOLD}Match Score:{Colors.END} {Colors.SCORE}{movie['similarity']:.2%}{Colors.END}")
        
        if 'genres' in movie and movie['genres']:
            print(f"\n🎭 {Colors.BOLD}Genres:{Colors.END} {movie['genres']}")
        
        if 'director' in movie and movie['director']:
            print(f"🎬 {Colors.BOLD}Director:{Colors.END} {movie['director']}")
        
        if 'actors' in movie and movie['actors']:
            actors_str = ", ".join([a for a in movie['actors'] if pd.notna(a)])
            if actors_str:
                print(f"⭐ {Colors.BOLD}Cast:{Colors.END} {actors_str}")
        
        if 'vote_average' in movie and movie['vote_average']:
            print(f"⭐ {Colors.BOLD}Rating:{Colors.END} {Colors.YELLOW}{movie['vote_average']}/10{Colors.END}")
        
        print(f"\n{Colors.CYAN}{'-' * 70}{Colors.END}")
        print(f"{Colors.BOLD}Actions:{Colors.END}")
        print(f"{Colors.YELLOW}1.{Colors.END} Find similar movies to this one")
        print(f"{Colors.YELLOW}2.{Colors.END} Back to recommendations list")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
        
        action = input(f"\n{Colors.YELLOW}Choose an action (1-2, or Enter to continue):{Colors.END} ").strip()
        
        if action == '1':
            # Find similar movies
            print(f"\n{Colors.INFO}Searching for movies similar to '{Colors.BOLD}{movie['title']}{Colors.END}{Colors.INFO}'...{Colors.END}")
            self.recommender.recommend_movies(movie['title'])
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
        elif action == '2':
            return
        else:
            return
    
    def analyze_sentiment_interactive(self):
        """Interactive flow for sentiment analysis"""
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}SENTIMENT ANALYSIS{Colors.END}")
        print(f"{Colors.HEADER}{'=' * 70}{Colors.END}")
        
        # Check if sentiment analysis is available
        if self.recommender.use_transformers:
            available = self.recommender.sentiment_pipeline is not None
        else:
            available = self.recommender.clf is not None
        
        if not available:
            print(f"\n{Colors.ERROR}❌ Sentiment analysis is not available (models not loaded){Colors.END}")
            return
        
        print(f"\n{Colors.INFO}Enter a movie review to analyze its sentiment.{Colors.END}")
        print(f"{Colors.INFO}(Type 'done' on a new line to finish, or press Enter twice){Colors.END}")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
        
        review_lines = []
        while True:
            line = input()
            if line.lower() == 'done':
                break
            if not line and review_lines:
                break
            if line:
                review_lines.append(line)
        
        review = " ".join(review_lines).strip()
        
        if not review:
            print(f"{Colors.ERROR}Error: Review cannot be empty!{Colors.END}")
            return
        
        print(f"\n{Colors.INFO}Analyzing sentiment...{Colors.END}")
        sentiment, confidence = self.recommender.analyze_sentiment(review)
        
        print(f"\n{Colors.CYAN}{'-' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}SENTIMENT ANALYSIS RESULTS{Colors.END}")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
        print(f"{Colors.BOLD}Review:{Colors.END} {review[:100]}{'...' if len(review) > 100 else ''}")
        
        # Color code sentiment
        if sentiment.lower() == 'positive':
            print(f"{Colors.BOLD}Sentiment:{Colors.END} {Colors.SUCCESS}{sentiment}{Colors.END}")
        else:
            print(f"{Colors.BOLD}Sentiment:{Colors.END} {Colors.ERROR}{sentiment}{Colors.END}")
        
        print(f"{Colors.BOLD}Confidence:{Colors.END} {Colors.YELLOW}{confidence * 100:.2f}%{Colors.END}")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
    
    def search_movies(self):
        """Interactive flow for searching movies"""
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}SEARCH MOVIES{Colors.END}")
        print(f"{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"\n{Colors.INFO}💡 Tip: Start typing and press Tab to autocomplete{Colors.END}")
        
        query = self.get_input_with_suggestions(f"\n{Colors.CYAN}Enter search query:{Colors.END} ", show_suggestions=True)
        
        if not query:
            print(f"{Colors.ERROR}Error: Search query cannot be empty!{Colors.END}")
            return
        
        print(f"\n{Colors.INFO}Searching for movies matching '{Colors.BOLD}{query}{Colors.END}{Colors.INFO}'...{Colors.END}")
        suggestions = self.recommender.get_movie_suggestions(query, limit=20)
        
        if not suggestions:
            print(f"\n{Colors.ERROR}❌ No movies found matching '{query}'{Colors.END}")
            return
        
        print(f"\n✓ Found {len(suggestions)} movies:\n")
        print("-" * 70)
        
        print(f"\n{Colors.SUCCESS}✓ Found {len(suggestions)} movies:{Colors.END}\n")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
        
        for i, movie in enumerate(suggestions, 1):
            print(f"{Colors.YELLOW}{i}.{Colors.END} {movie}")
        
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
    
    def view_movie_details(self):
        """Interactive flow for viewing movie details"""
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}MOVIE DETAILS{Colors.END}")
        print(f"{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"\n{Colors.INFO}💡 Tip: Start typing and press Tab to autocomplete{Colors.END}")
        
        movie_title = self.get_input_with_suggestions(f"\n{Colors.CYAN}Enter movie title:{Colors.END} ", show_suggestions=True)
        
        if not movie_title:
            print(f"{Colors.ERROR}Error: Movie title cannot be empty!{Colors.END}")
            return
        
        movie_info = self.recommender.get_movie_info(movie_title)
        
        if not movie_info:
            print(f"\n{Colors.ERROR}❌ Movie '{movie_title}' not found in database{Colors.END}")
            
            # Suggest similar movies
            suggestions = self.recommender.get_movie_suggestions(movie_title, limit=5)
            if suggestions:
                print(f"\n{Colors.WARNING}💡 Did you mean one of these?{Colors.END}")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"   {Colors.YELLOW}{i}.{Colors.END} {suggestion}")
            return
        
        print(f"\n{Colors.CYAN}{'-' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}MOVIE INFORMATION{Colors.END}")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
        print(f"{Colors.BOLD}Title:{Colors.END} {Colors.HIGHLIGHT}{movie_info['title']}{Colors.END}")
        print(f"{Colors.BOLD}Director:{Colors.END} {movie_info['director']}")
        print(f"{Colors.BOLD}Genres:{Colors.END} {movie_info['genres']}")
        actors_str = ", ".join([a for a in movie_info['actors'] if pd.notna(a)])
        print(f"{Colors.BOLD}Actors:{Colors.END} {actors_str}")
        print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
    
    def show_evaluation_metrics(self):
        """Display evaluation metrics for both systems"""
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}MODEL EVALUATION METRICS{Colors.END}")
        print(f"{Colors.HEADER}{'=' * 70}{Colors.END}")
        
        try:
            from src.evaluator import ModelEvaluator
            evaluator = ModelEvaluator(self.recommender)
            
            print(f"\n{Colors.INFO}Evaluating models... This may take a moment.{Colors.END}\n")
            
            # Evaluate sentiment
            print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
            print(f"{Colors.BOLD}{Colors.BLUE}SENTIMENT ANALYSIS METRICS{Colors.END}")
            print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
            
            sentiment_metrics = evaluator.evaluate_sentiment()
            
            if 'error' in sentiment_metrics:
                print(f"{Colors.ERROR}❌ {sentiment_metrics['error']}{Colors.END}")
            else:
                print(f"{Colors.BOLD}Model Type:{Colors.END} {sentiment_metrics['model_type']}")
                print(f"{Colors.BOLD}Total Test Samples:{Colors.END} {sentiment_metrics['total_samples']}")
                print(f"\n{Colors.BOLD}Overall Metrics:{Colors.END}")
                print(f"  {Colors.SUCCESS}Accuracy:{Colors.END} {sentiment_metrics['accuracy']:.2f}%")
                print(f"  {Colors.SUCCESS}Macro Precision:{Colors.END} {sentiment_metrics['macro_precision']:.2f}%")
                print(f"  {Colors.SUCCESS}Macro Recall:{Colors.END} {sentiment_metrics['macro_recall']:.2f}%")
                print(f"  {Colors.SUCCESS}Macro F1 Score:{Colors.END} {sentiment_metrics['macro_f1']:.2f}%")
                
                print(f"\n{Colors.BOLD}Positive Class:{Colors.END}")
                print(f"  Precision: {sentiment_metrics['precision_positive']:.2f}%")
                print(f"  Recall: {sentiment_metrics['recall_positive']:.2f}%")
                print(f"  F1 Score: {sentiment_metrics['f1_positive']:.2f}%")
                
                print(f"\n{Colors.BOLD}Negative Class:{Colors.END}")
                print(f"  Precision: {sentiment_metrics['precision_negative']:.2f}%")
                print(f"  Recall: {sentiment_metrics['recall_negative']:.2f}%")
                print(f"  F1 Score: {sentiment_metrics['f1_negative']:.2f}%")
                
                cm = sentiment_metrics['confusion_matrix']
                print(f"\n{Colors.BOLD}Confusion Matrix:{Colors.END}")
                print(f"  True Positives: {cm['true_positive']}")
                print(f"  False Positives: {cm['false_positive']}")
                print(f"  False Negatives: {cm['false_negative']}")
                print(f"  True Negatives: {cm['true_negative']}")
            
            # Evaluate recommendations
            print(f"\n{Colors.CYAN}{'-' * 70}{Colors.END}")
            print(f"{Colors.BOLD}{Colors.BLUE}RECOMMENDATION SYSTEM METRICS{Colors.END}")
            print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
            
            rec_metrics = evaluator.evaluate_recommendations()
            
            if 'error' in rec_metrics:
                print(f"{Colors.ERROR}❌ {rec_metrics['error']}{Colors.END}")
            else:
                print(f"{Colors.BOLD}Model Type:{Colors.END} {rec_metrics['model_type']}")
                print(f"{Colors.BOLD}Test Movies:{Colors.END} {rec_metrics['num_test_movies']}")
                print(f"{Colors.BOLD}Recommendations per Movie (K):{Colors.END} {rec_metrics['k']}")
                print(f"\n{Colors.BOLD}Performance Metrics:{Colors.END}")
                print(f"  {Colors.SUCCESS}Precision@{rec_metrics['k']}:{Colors.END} {rec_metrics['precision_at_k']:.2f}%")
                print(f"  {Colors.SUCCESS}Recall@{rec_metrics['k']}:{Colors.END} {rec_metrics['recall_at_k']:.2f}%")
                print(f"  {Colors.SUCCESS}F1@{rec_metrics['k']}:{Colors.END} {rec_metrics['f1_at_k']:.2f}%")
                print(f"  {Colors.SUCCESS}Diversity:{Colors.END} {rec_metrics['diversity']:.2f}%")
                print(f"  {Colors.SUCCESS}Coverage:{Colors.END} {rec_metrics['coverage']:.2f}%")
                print(f"  {Colors.INFO}Total Movies in Database:{Colors.END} {rec_metrics['total_movies']}")
            
            print(f"\n{Colors.CYAN}{'-' * 70}{Colors.END}")
            
        except ImportError:
            print(f"{Colors.ERROR}❌ Evaluation module not found{Colors.END}")
        except Exception as e:
            print(f"{Colors.ERROR}❌ Error during evaluation: {e}{Colors.END}")
    
    def run(self):
        """Main application loop"""
        self.clear_screen()
        self.print_header()
        
        while True:
            self.print_menu()
            choice = input(f"\n{Colors.YELLOW}Enter your choice (1-6):{Colors.END} ").strip()
            
            if choice == '1':
                self.get_movie_recommendations()
            elif choice == '2':
                self.analyze_sentiment_interactive()
            elif choice == '3':
                self.search_movies()
            elif choice == '4':
                self.view_movie_details()
            elif choice == '5':
                self.show_evaluation_metrics()
            elif choice == '6':
                print(f"\n{Colors.TITLE}{'=' * 70}{Colors.END}")
                print(f"{Colors.SUCCESS}Thank you for using the Movie Recommendation System!{Colors.END}")
                print(f"{Colors.TITLE}{'=' * 70}{Colors.END}")
                break
            else:
                print(f"\n{Colors.ERROR}❌ Invalid choice! Please enter a number between 1 and 6.{Colors.END}")
            
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")


def main():
    """Main entry point"""
    try:
        # Initialize recommender system
        recommender = MovieRecommender()
        
        # Initialize and run terminal interface
        interface = TerminalInterface(recommender)
        interface.run()
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Program interrupted by user. Goodbye!{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.ERROR}❌ An error occurred: {e}{Colors.END}")
        sys.exit(1)


if __name__ == '__main__':
    main()

