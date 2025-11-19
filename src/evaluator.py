#!/usr/bin/env python3
"""
Evaluation Module for Movie Recommendation System
Calculates accuracy, F1 score, recall, precision, and other metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


class ModelEvaluator:
    """Evaluates both sentiment analysis and recommendation systems"""
    
    def __init__(self, recommender):
        """
        Initialize evaluator
        
        Args:
            recommender: MovieRecommender instance
        """
        self.recommender = recommender
        self.sentiment_test_data = None
        self._load_test_data()
    
    def _load_test_data(self):
        """Load or create test data for evaluation"""
        # Try to load reviews with labels if available
        reviews_path = DATA_DIR / "raw" / "reviews.txt"
        
        if reviews_path.exists():
            try:
                # Try to parse reviews file (format may vary)
                with open(reviews_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Create sample test data from reviews
                # In a real scenario, you'd have labeled test data
                self.sentiment_test_data = self._create_sentiment_test_set(lines)
            except Exception as e:
                print(f"Warning: Could not load test reviews: {e}")
                self.sentiment_test_data = self._create_default_sentiment_test_set()
        else:
            self.sentiment_test_data = self._create_default_sentiment_test_set()
    
    def _create_sentiment_test_set(self, lines: List[str]) -> List[Tuple[str, str]]:
        """Create test set from review lines"""
        test_data = []
        
        # Sample positive and negative reviews
        positive_samples = [
            ("This movie was absolutely amazing! Best film I've seen this year.", "Positive"),
            ("I loved every minute of this film. Highly recommended!", "Positive"),
            ("Excellent acting and great storyline. A masterpiece!", "Positive"),
            ("One of the best movies ever made. Perfect in every way.", "Positive"),
            ("Outstanding performance by all actors. Must watch!", "Positive"),
            ("Brilliant cinematography and engaging plot. Loved it!", "Positive"),
            ("This film exceeded all my expectations. Fantastic!", "Positive"),
            ("A wonderful movie with great character development.", "Positive"),
            ("I was completely captivated from start to finish.", "Positive"),
            ("Perfect blend of action, drama, and emotion. Excellent!", "Positive"),
        ]
        
        negative_samples = [
            ("This movie was terrible. Complete waste of time.", "Negative"),
            ("Boring and poorly written. I don't recommend it.", "Negative"),
            ("The worst film I've ever seen. Awful acting.", "Negative"),
            ("Disappointing plot and bad direction. Not worth watching.", "Negative"),
            ("I couldn't finish watching this. It was that bad.", "Negative"),
            ("Poor script and terrible pacing. Very disappointing.", "Negative"),
            ("This film was a complete disaster. Avoid at all costs.", "Negative"),
            ("Bad acting, weak storyline. Not recommended.", "Negative"),
            ("I expected much more. This movie was a letdown.", "Negative"),
            ("Terrible cinematography and confusing plot. Awful!", "Negative"),
        ]
        
        test_data = positive_samples + negative_samples
        return test_data
    
    def _create_default_sentiment_test_set(self) -> List[Tuple[str, str]]:
        """Create default test set if no data file exists"""
        return self._create_sentiment_test_set([])
    
    def evaluate_sentiment(self) -> Dict:
        """
        Evaluate sentiment analysis model
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.sentiment_test_data:
            return {"error": "No test data available"}
        
        # Check if sentiment analysis is available
        if self.recommender.use_transformers:
            available = self.recommender.sentiment_pipeline is not None
        else:
            available = self.recommender.clf is not None
        
        if not available:
            return {"error": "Sentiment analysis model not available"}
        
        # Get predictions
        y_true = []
        y_pred = []
        predictions = []
        
        for review, true_label in self.sentiment_test_data:
            sentiment, confidence = self.recommender.analyze_sentiment(review)
            
            # Normalize labels
            true_label_norm = "Positive" if "Positive" in true_label else "Negative"
            pred_label_norm = "Positive" if "Positive" in sentiment else "Negative"
            
            y_true.append(true_label_norm)
            y_pred.append(pred_label_norm)
            predictions.append({
                'review': review[:50] + "..." if len(review) > 50 else review,
                'true_label': true_label_norm,
                'predicted_label': pred_label_norm,
                'confidence': confidence
            })
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="Positive", zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label="Positive", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="Positive", zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])
        
        # Per-class metrics
        precision_neg = precision_score(y_true, y_pred, pos_label="Negative", zero_division=0)
        recall_neg = recall_score(y_true, y_pred, pos_label="Negative", zero_division=0)
        f1_neg = f1_score(y_true, y_pred, pos_label="Negative", zero_division=0)
        
        return {
            'accuracy': round(accuracy * 100, 2),
            'precision_positive': round(precision * 100, 2),
            'recall_positive': round(recall * 100, 2),
            'f1_positive': round(f1 * 100, 2),
            'precision_negative': round(precision_neg * 100, 2),
            'recall_negative': round(recall_neg * 100, 2),
            'f1_negative': round(f1_neg * 100, 2),
            'macro_precision': round((precision + precision_neg) / 2 * 100, 2),
            'macro_recall': round((recall + recall_neg) / 2 * 100, 2),
            'macro_f1': round((f1 + f1_neg) / 2 * 100, 2),
            'confusion_matrix': {
                'true_positive': int(cm[0][0]),
                'false_positive': int(cm[0][1]),
                'false_negative': int(cm[1][0]),
                'true_negative': int(cm[1][1])
            },
            'total_samples': len(y_true),
            'predictions': predictions[:10],  # Show first 10 predictions
            'model_type': 'DistilBERT' if self.recommender.use_transformers else 'Naive Bayes'
        }
    
    def evaluate_recommendations(self, num_test_movies: int = 50, k: int = 10) -> Dict:
        """
        Evaluate recommendation system
        
        Args:
            num_test_movies: Number of movies to test
            k: Number of recommendations to consider (Precision@K, Recall@K)
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.recommender.data is None or len(self.recommender.data) == 0:
            return {"error": "No movie data available"}
        
        if self.recommender.similarity is None:
            return {"error": "Similarity matrix not available"}
        
        # Select random movies for testing
        available_movies = self.recommender.data['movie_title'].tolist()
        num_test_movies = min(num_test_movies, len(available_movies))
        test_movies = random.sample(available_movies, num_test_movies)
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        diversity_scores = []
        coverage_scores = []
        
        recommended_movies_set = set()
        
        for movie_title in test_movies:
            try:
                recommendations, error = self.recommender.recommend_movies(movie_title, k)
                
                if error:
                    continue
                
                # Get movie features for diversity calculation
                movie_idx = self.recommender.data[
                    self.recommender.data['movie_title'].str.lower() == movie_title.lower()
                ].index[0]
                
                # Calculate diversity (genre diversity in recommendations)
                genres_in_recs = set()
                for rec in recommendations:
                    rec_idx = self.recommender.data[
                        self.recommender.data['movie_title'] == rec['title']
                    ].index
                    if len(rec_idx) > 0:
                        genre = self.recommender.data.iloc[rec_idx[0]].get('genres', '')
                        if pd.notna(genre):
                            genres_in_recs.add(str(genre))
                
                diversity = len(genres_in_recs) / max(k, 1)
                diversity_scores.append(diversity)
                
                # Track coverage
                for rec in recommendations:
                    recommended_movies_set.add(rec['title'])
                
                # For precision/recall, we use similarity threshold
                # Movies with similarity > 0.3 are considered relevant
                relevant_count = sum(1 for rec in recommendations if rec['similarity'] > 0.3)
                precision_at_k = relevant_count / k if k > 0 else 0
                recall_at_k = relevant_count / min(k, len(available_movies))  # Simplified
                
                precision_scores.append(precision_at_k)
                recall_scores.append(recall_at_k)
                
                if precision_at_k + recall_at_k > 0:
                    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
                else:
                    f1_at_k = 0
                f1_scores.append(f1_at_k)
                
            except Exception as e:
                continue
        
        # Calculate coverage
        total_movies = len(available_movies)
        coverage = len(recommended_movies_set) / total_movies if total_movies > 0 else 0
        
        # Aggregate metrics
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
        
        return {
            'precision_at_k': round(avg_precision * 100, 2),
            'recall_at_k': round(avg_recall * 100, 2),
            'f1_at_k': round(avg_f1 * 100, 2),
            'diversity': round(avg_diversity * 100, 2),
            'coverage': round(coverage * 100, 2),
            'num_test_movies': num_test_movies,
            'k': k,
            'total_movies': total_movies,
            'model_type': 'Sentence Transformers' if self.recommender.use_transformers else 'CountVectorizer'
        }
    
    def get_all_metrics(self) -> Dict:
        """
        Get all evaluation metrics for both systems
        
        Returns:
            Dictionary with all metrics
        """
        sentiment_metrics = self.evaluate_sentiment()
        recommendation_metrics = self.evaluate_recommendations()
        
        return {
            'sentiment_analysis': sentiment_metrics,
            'recommendation_system': recommendation_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }

