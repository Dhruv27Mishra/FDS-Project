#!/usr/bin/env python3
"""
Flask Web Application for Multi-Algorithm Sentiment Analysis System
"""

from flask import Flask, render_template, request, jsonify
from src.sentiment_analyzer import SentimentAnalyzer
from src.algorithm_comparison import AlgorithmComparator
import sys

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Initialize the sentiment analysis system
print("Initializing Sentiment Analysis System...")
try:
    analyzer = SentimentAnalyzer()
    comparator = AlgorithmComparator(analyzer)
    available_algorithms = analyzer.get_available_algorithms()
    print(f"✓ System ready! {len(available_algorithms)} algorithm(s) loaded")
    for alg in available_algorithms:
        print(f"  - {analyzer.algorithms[alg]['name']}")
except Exception as e:
    print(f"❌ Error initializing system: {e}")
    sys.exit(1)


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """API endpoint for getting available algorithms"""
    algorithms = []
    for alg in analyzer.get_available_algorithms():
        algorithms.append({
            'id': alg,
            'name': analyzer.algorithms[alg]['name']
        })
    return jsonify({'algorithms': algorithms})


@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """API endpoint for sentiment analysis"""
    data = request.get_json()
    review = data.get('review', '').strip()
    algorithm = data.get('algorithm', 'distilbert')
    
    if not review:
        return jsonify({'error': 'Review text is required'}), 400
    
    sentiment, confidence, metadata = analyzer.analyze(review, algorithm)
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence,
        'confidence_percent': round(confidence * 100, 2),
        'algorithm': analyzer.algorithms[algorithm]['name'],
        'metadata': metadata
    })


@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """API endpoint for comparing all algorithms"""
    data = request.get_json()
    review = data.get('review', '').strip()
    
    if not review:
        return jsonify({'error': 'Review text is required'}), 400
    
    results = analyzer.compare_algorithms(review)
    
    return jsonify({
        'results': results,
        'text': review[:100] + '...' if len(review) > 100 else review
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """API endpoint for getting algorithm performance metrics"""
    try:
        results = comparator.evaluate_all_algorithms()
        chart_base64 = comparator.create_comparison_chart(results)
        
        return jsonify({
            'metrics': results,
            'chart': chart_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Starting Sentiment Analysis Web Application...")
    print("=" * 70)
    print("\nServer will be available at: http://localhost:5002")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5002, host='0.0.0.0')
