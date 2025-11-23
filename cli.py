#!/usr/bin/env python3
"""
CLI Interface for Multi-Algorithm Sentiment Analysis System
"""

from src.sentiment_analyzer import SentimentAnalyzer
from src.algorithm_comparison import AlgorithmComparator
import sys

def main():
    """Main entry point for CLI"""
    try:
        print("=" * 70)
        print(" " * 20 + "SENTIMENT ANALYSIS SYSTEM")
        print(" " * 15 + "Multi-Algorithm Comparison Tool")
        print("=" * 70)
        print()
        
        # Initialize analyzer
        print("Initializing sentiment analysis algorithms...")
        analyzer = SentimentAnalyzer()
        comparator = AlgorithmComparator(analyzer)
        
        available = analyzer.get_available_algorithms()
        print(f"✓ Loaded {len(available)} algorithm(s): {', '.join([analyzer.algorithms[alg]['name'] for alg in available])}")
        print()
        
        while True:
            print("-" * 70)
            print("MAIN MENU")
            print("-" * 70)
            print("1. Analyze Sentiment (Single Algorithm)")
            print("2. Compare All Algorithms")
            print("3. View Algorithm Performance Metrics")
            print("4. Exit")
            print("-" * 70)
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                analyze_sentiment(analyzer)
            elif choice == '2':
                compare_algorithms(analyzer)
            elif choice == '3':
                view_performance_metrics(comparator)
            elif choice == '4':
                print("\nThank you for using the Sentiment Analysis System!")
                break
            else:
                print("\n❌ Invalid choice! Please enter a number between 1 and 4.")
            
            input("\nPress Enter to continue...")
            print()
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)


def analyze_sentiment(analyzer: SentimentAnalyzer):
    """Analyze sentiment with a single algorithm"""
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS")
    print("=" * 70)
    
    available = analyzer.get_available_algorithms()
    if not available:
        print("❌ No algorithms available!")
        return
    
    print("\nAvailable algorithms:")
    for i, alg in enumerate(available, 1):
        print(f"  {i}. {analyzer.algorithms[alg]['name']}")
    
    try:
        alg_choice = input(f"\nSelect algorithm (1-{len(available)}): ").strip()
        alg_index = int(alg_choice) - 1
        if 0 <= alg_index < len(available):
            selected_alg = available[alg_index]
        else:
            print("❌ Invalid selection!")
            return
    except ValueError:
        print("❌ Invalid input!")
        return
    
    print("\nEnter text to analyze (type 'done' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line.lower() == 'done':
            break
        if line:
            lines.append(line)
    
    text = " ".join(lines).strip()
    
    if not text:
        print("❌ No text provided!")
        return
    
    print(f"\nAnalyzing with {analyzer.algorithms[selected_alg]['name']}...")
    sentiment, confidence, metadata = analyzer.analyze(text, selected_alg)
    
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence * 100:.2f}%")
    if metadata:
        print(f"Details: {metadata}")
    print("-" * 70)


def compare_algorithms(analyzer: SentimentAnalyzer):
    """Compare all algorithms on the same text"""
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)
    
    print("\nEnter text to analyze (type 'done' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line.lower() == 'done':
            break
        if line:
            lines.append(line)
    
    text = " ".join(lines).strip()
    
    if not text:
        print("❌ No text provided!")
        return
    
    print("\nComparing all algorithms...")
    results = analyzer.compare_algorithms(text)
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    for alg, result in results.items():
        print(f"\n{result['name']}:")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Confidence: {result['confidence'] * 100:.2f}%")
    
    print("\n" + "=" * 70)


def view_performance_metrics(comparator: AlgorithmComparator):
    """View performance metrics for all algorithms"""
    print("\n" + "=" * 70)
    print("ALGORITHM PERFORMANCE METRICS")
    print("=" * 70)
    print("\nEvaluating algorithms on test dataset...")
    
    results = comparator.evaluate_all_algorithms()
    
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    
    for alg, metrics in results.items():
        if 'error' in metrics:
            print(f"\n{metrics.get('name', alg)}: Error - {metrics['error']}")
            continue
        
        print(f"\n{metrics['name']}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.2f}%")
        print(f"  Recall: {metrics['recall']:.2f}%")
        print(f"  F1 Score: {metrics['f1_score']:.2f}%")
        print(f"  Avg Confidence: {metrics['avg_confidence']:.2f}%")
        print(f"  Test Samples: {metrics['total_samples']}")
        
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TP: {cm['true_positive']}, FP: {cm['false_positive']}")
        print(f"    FN: {cm['false_negative']}, TN: {cm['true_negative']}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
