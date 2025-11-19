#!/usr/bin/env python3
"""
Flask Web Application for Movie Recommendation System with Sentiment Analysis
"""

from flask import Flask, render_template, request, jsonify
from src.movie_recommender import MovieRecommender
from src.evaluator import ModelEvaluator
import sys

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Initialize the recommender system
print("Initializing Movie Recommendation System...")
try:
    recommender = MovieRecommender()
    print("✓ System ready!")
except Exception as e:
    print(f"❌ Error initializing system: {e}")
    sys.exit(1)


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/search', methods=['GET'])
def search_movies():
    """API endpoint for searching movies"""
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    suggestions = recommender.get_movie_suggestions(query, limit=limit)
    return jsonify({'movies': suggestions})


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint for getting movie recommendations"""
    data = request.get_json()
    movie_title = data.get('movie_title', '').strip()
    num_recommendations = int(data.get('num_recommendations', 10))
    
    if not movie_title:
        return jsonify({'error': 'Movie title is required'}), 400
    
    recommendations, error = recommender.recommend_movies(movie_title, num_recommendations)
    
    if error:
        return jsonify({'error': error}), 404
    
    return jsonify({
        'recommendations': recommendations,
        'count': len(recommendations)
    })


@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    """API endpoint for sentiment analysis"""
    data = request.get_json()
    review = data.get('review', '').strip()
    
    if not review:
        return jsonify({'error': 'Review text is required'}), 400
    
    sentiment, confidence = recommender.analyze_sentiment(review)
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence,
        'confidence_percent': round(confidence * 100, 2)
    })


@app.route('/api/movie-info', methods=['GET'])
def get_movie_info():
    """API endpoint for getting movie information"""
    movie_title = request.args.get('title', '').strip()
    
    if not movie_title:
        return jsonify({'error': 'Movie title is required'}), 400
    
    movie_info = recommender.get_movie_info(movie_title)
    
    if not movie_info:
        return jsonify({'error': f'Movie "{movie_title}" not found'}), 404
    
    return jsonify(movie_info)


@app.route('/api/evaluate', methods=['GET'])
def evaluate_models():
    """API endpoint for evaluating model performance"""
    try:
        evaluator = ModelEvaluator(recommender)
        metrics = evaluator.get_all_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Starting Movie Recommendation Web Application...")
    print("=" * 70)
    print("\nServer will be available at: http://localhost:5002")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5002, host='0.0.0.0')

