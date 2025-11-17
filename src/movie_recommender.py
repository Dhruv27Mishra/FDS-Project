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
from typing import List, Tuple, Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DATA_PATH = DATA_DIR / "main_data.csv"
SENTIMENT_MODEL_PATH = MODELS_DIR / "nlp_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tranform.pkl"


class MovieRecommender:
    """Main class for the Movie Recommendation System with Sentiment Analysis"""
    
    def __init__(self):
        """Initialize the recommender system"""
        self.data = None
        self.similarity = None
        self.clf = None
        self.vectorizer = None
        self._load_data()
        self._load_models()
    
    def _load_data(self):
        """Load movie data and create similarity matrix"""
        print("Loading movie data...")
        try:
            self.data = pd.read_csv(PROCESSED_DATA_PATH)
            print(f"✓ Loaded {len(self.data)} movies")
            
            # Create similarity matrix
            print("Computing similarity matrix...")
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(self.data['comb'])
            self.similarity = cosine_similarity(count_matrix)
            print("✓ Similarity matrix computed")
        except FileNotFoundError:
            print(f"Error: {PROCESSED_DATA_PATH} not found!")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def _load_models(self):
        """Load pre-trained sentiment analysis models"""
        print("Loading sentiment analysis models...")
        try:
            with open(SENTIMENT_MODEL_PATH, 'rb') as model_file:
                self.clf = pickle.load(model_file)
            with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
                self.vectorizer = pickle.load(vectorizer_file)
            print("✓ Sentiment analysis models loaded")
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
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print application header"""
        print("=" * 70)
        print(" " * 15 + "MOVIE RECOMMENDATION SYSTEM")
        print(" " * 10 + "with Sentiment Analysis")
        print("=" * 70)
        print()
    
    def print_menu(self):
        """Print main menu options"""
        print("\n" + "-" * 70)
        print("MAIN MENU")
        print("-" * 70)
        print("1. Get Movie Recommendations")
        print("2. Analyze Review Sentiment")
        print("3. Search Movies")
        print("4. View Movie Details")
        print("5. Exit")
        print("-" * 70)
    
    def get_movie_recommendations(self):
        """Interactive flow for getting movie recommendations"""
        print("\n" + "=" * 70)
        print("MOVIE RECOMMENDATIONS")
        print("=" * 70)
        
        movie_title = input("\nEnter a movie title: ").strip()
        
        if not movie_title:
            print("Error: Movie title cannot be empty!")
            return
        
        print(f"\nSearching for recommendations based on '{movie_title}'...")
        recommendations, error = self.recommender.recommend_movies(movie_title)
        
        if error:
            print(f"\n❌ {error}")
            return
        
        print(f"\n✓ Found {len(recommendations)} recommendations:\n")
        print("-" * 70)
        
        for i, movie in enumerate(recommendations, 1):
            print(f"\n{i}. {movie['title']}")
            print(f"   Similarity Score: {movie['similarity']:.4f}")
            print(f"   Director: {movie['director']}")
            print(f"   Genres: {movie['genres']}")
            actors_str = ", ".join([a for a in movie['actors'] if pd.notna(a)])
            print(f"   Actors: {actors_str}")
        
        print("\n" + "-" * 70)
    
    def analyze_sentiment_interactive(self):
        """Interactive flow for sentiment analysis"""
        print("\n" + "=" * 70)
        print("SENTIMENT ANALYSIS")
        print("=" * 70)
        
        if self.recommender.clf is None:
            print("\n❌ Sentiment analysis is not available (models not loaded)")
            return
        
        print("\nEnter a movie review to analyze its sentiment.")
        print("(Type 'done' on a new line to finish, or press Enter twice)")
        print("-" * 70)
        
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
            print("Error: Review cannot be empty!")
            return
        
        print("\nAnalyzing sentiment...")
        sentiment, confidence = self.recommender.analyze_sentiment(review)
        
        print("\n" + "-" * 70)
        print("SENTIMENT ANALYSIS RESULTS")
        print("-" * 70)
        print(f"Review: {review[:100]}{'...' if len(review) > 100 else ''}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("-" * 70)
    
    def search_movies(self):
        """Interactive flow for searching movies"""
        print("\n" + "=" * 70)
        print("SEARCH MOVIES")
        print("=" * 70)
        
        query = input("\nEnter search query: ").strip()
        
        if not query:
            print("Error: Search query cannot be empty!")
            return
        
        print(f"\nSearching for movies matching '{query}'...")
        suggestions = self.recommender.get_movie_suggestions(query, limit=20)
        
        if not suggestions:
            print(f"\n❌ No movies found matching '{query}'")
            return
        
        print(f"\n✓ Found {len(suggestions)} movies:\n")
        print("-" * 70)
        
        for i, movie in enumerate(suggestions, 1):
            print(f"{i}. {movie}")
        
        print("-" * 70)
    
    def view_movie_details(self):
        """Interactive flow for viewing movie details"""
        print("\n" + "=" * 70)
        print("MOVIE DETAILS")
        print("=" * 70)
        
        movie_title = input("\nEnter movie title: ").strip()
        
        if not movie_title:
            print("Error: Movie title cannot be empty!")
            return
        
        movie_info = self.recommender.get_movie_info(movie_title)
        
        if not movie_info:
            print(f"\n❌ Movie '{movie_title}' not found in database")
            return
        
        print("\n" + "-" * 70)
        print("MOVIE INFORMATION")
        print("-" * 70)
        print(f"Title: {movie_info['title']}")
        print(f"Director: {movie_info['director']}")
        print(f"Genres: {movie_info['genres']}")
        actors_str = ", ".join([a for a in movie_info['actors'] if pd.notna(a)])
        print(f"Actors: {actors_str}")
        print("-" * 70)
    
    def run(self):
        """Main application loop"""
        self.clear_screen()
        self.print_header()
        
        while True:
            self.print_menu()
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.get_movie_recommendations()
            elif choice == '2':
                self.analyze_sentiment_interactive()
            elif choice == '3':
                self.search_movies()
            elif choice == '4':
                self.view_movie_details()
            elif choice == '5':
                print("\n" + "=" * 70)
                print("Thank you for using the Movie Recommendation System!")
                print("=" * 70)
                break
            else:
                print("\n❌ Invalid choice! Please enter a number between 1 and 5.")
            
            input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    try:
        # Initialize recommender system
        recommender = MovieRecommender()
        
        # Initialize and run terminal interface
        interface = TerminalInterface(recommender)
        interface.run()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

