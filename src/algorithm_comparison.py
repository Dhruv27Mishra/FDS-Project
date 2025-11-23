#!/usr/bin/env python3
"""
Algorithm Comparison and Visualization Module
Compares performance of different sentiment analysis algorithms
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from math import pi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from .test_reviews import get_test_reviews

BASE_DIR = Path(__file__).resolve().parents[1]


class AlgorithmComparator:
    """Compare and visualize sentiment analysis algorithms"""
    
    def __init__(self, sentiment_analyzer):
        """
        Initialize comparator
        
        Args:
            sentiment_analyzer: SentimentAnalyzer instance
        """
        self.analyzer = sentiment_analyzer
        self.test_data = self._create_test_dataset()
    
    def _create_test_dataset(self) -> List[Tuple[str, str]]:
        """Create test dataset for evaluation using comprehensive test reviews"""
        return get_test_reviews()
    
    def evaluate_all_algorithms(self) -> Dict:
        """
        Evaluate all available algorithms on test dataset
        
        Returns:
            Dictionary with evaluation metrics for each algorithm
        """
        results = {}
        available_algorithms = self.analyzer.get_available_algorithms()
        
        for algorithm in available_algorithms:
            y_true = []
            y_pred = []
            confidences = []
            
            for text, true_label in self.test_data:
                sentiment, confidence, _ = self.analyzer.analyze(text, algorithm)
                
                # Normalize labels
                true_label_norm = true_label
                pred_label_norm = sentiment
                
                # For binary classification, convert Neutral to Positive
                if true_label_norm == "Neutral":
                    true_label_norm = "Positive"
                if pred_label_norm == "Neutral":
                    pred_label_norm = "Positive"
                
                y_true.append(true_label_norm)
                y_pred.append(pred_label_norm)
                confidences.append(confidence)
            
            # Calculate metrics
            try:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, pos_label="Positive", zero_division=0, average='binary')
                recall = recall_score(y_true, y_pred, pos_label="Positive", zero_division=0, average='binary')
                f1 = f1_score(y_true, y_pred, pos_label="Positive", zero_division=0, average='binary')
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])
                
                results[algorithm] = {
                    'name': self.analyzer.algorithms[algorithm]['name'],
                    'accuracy': round(accuracy * 100, 2),
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2),
                    'f1_score': round(f1 * 100, 2),
                    'avg_confidence': round(np.mean(confidences) * 100, 2),
                    'confusion_matrix': {
                    'true_positive': int(cm[0][0]) if len(cm) > 0 and len(cm[0]) > 0 else 0,
                    'false_positive': int(cm[0][1]) if len(cm) > 0 and len(cm[0]) > 1 else 0,
                    'false_negative': int(cm[1][0]) if len(cm) > 1 and len(cm[1]) > 0 else 0,
                    'true_negative': int(cm[1][1]) if len(cm) > 1 and len(cm[1]) > 1 else 0
                },
                'total_samples': len(y_true)
            }
            except Exception as e:
                results[algorithm] = {'error': str(e)}
        
        return results
    
    def create_all_charts(self, results: Dict) -> Dict[str, str]:
        """
        Create multiple types of comparison charts
        
        Args:
            results: Evaluation results from evaluate_all_algorithms()
        
        Returns:
            Dictionary with different chart types as base64 encoded images
        """
        # Filter out algorithms with errors
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {}
        
        algorithms = [v['name'] for v in valid_results.values()]
        accuracies = [v['accuracy'] for v in valid_results.values()]
        precisions = [v['precision'] for v in valid_results.values()]
        recalls = [v['recall'] for v in valid_results.values()]
        f1_scores = [v['f1_score'] for v in valid_results.values()]
        confidences = [v['avg_confidence'] for v in valid_results.values()]
        
        charts = {}
        
        # 1. Grouped Bar Chart - All Metrics Comparison
        charts['grouped_bar'] = self._create_grouped_bar_chart(algorithms, accuracies, precisions, recalls, f1_scores)
        
        # 2. Line Chart - Metrics Trend
        charts['line'] = self._create_line_chart(algorithms, accuracies, precisions, recalls, f1_scores)
        
        # 3. Radar/Spider Chart - Multi-metric Comparison
        charts['radar'] = self._create_radar_chart(algorithms, accuracies, precisions, recalls, f1_scores, confidences)
        
        # 4. AI Summary and Recommendations (will be displayed as text, not image)
        # Skip image generation for recommendations
        
        # 5. Scatter Plot - Accuracy vs Confidence
        charts['scatter'] = self._create_scatter_plot(valid_results)
        
        # 6. Pie Chart - Best Performing Algorithms
        charts['pie'] = self._create_pie_chart(valid_results)
        
        # 7. Horizontal Bar Chart - Overall Ranking
        charts['horizontal_bar'] = self._create_horizontal_bar_chart(algorithms, accuracies, precisions, recalls, f1_scores)
        
        return charts
    
    def _create_grouped_bar_chart(self, algorithms, accuracies, precisions, recalls, f1_scores) -> str:
        """Create grouped bar chart comparing all metrics"""
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(algorithms))
        width = 0.2
        
        ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#4CAF50', alpha=0.8)
        ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#2196F3', alpha=0.8)
        ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#FF9800', alpha=0.8)
        ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color='#9C27B0', alpha=0.8)
        
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('All Metrics Comparison - Grouped Bar Chart', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_line_chart(self, algorithms, accuracies, precisions, recalls, f1_scores) -> str:
        """Create line chart showing metrics trends"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(algorithms))
        ax.plot(x, accuracies, marker='o', label='Accuracy', linewidth=2, color='#4CAF50', markersize=8)
        ax.plot(x, precisions, marker='s', label='Precision', linewidth=2, color='#2196F3', markersize=8)
        ax.plot(x, recalls, marker='^', label='Recall', linewidth=2, color='#FF9800', markersize=8)
        ax.plot(x, f1_scores, marker='d', label='F1 Score', linewidth=2, color='#9C27B0', markersize=8)
        
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Metrics Trend - Line Chart', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_radar_chart(self, algorithms, accuracies, precisions, recalls, f1_scores, confidences) -> str:
        """Create radar/spider chart for multi-metric comparison"""
        try:
            # Select top 5 algorithms for readability
            top_indices = sorted(range(len(accuracies)), key=lambda i: accuracies[i], reverse=True)[:5]
            
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confidence']
            N = len(categories)
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
            
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
            
            for idx, alg_idx in enumerate(top_indices):
                values = [
                    accuracies[alg_idx],
                    precisions[alg_idx],
                    recalls[alg_idx],
                    f1_scores[alg_idx],
                    confidences[alg_idx]
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=algorithms[alg_idx], color=colors[idx % len(colors)])
                ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.set_title('Multi-Metric Comparison - Radar Chart', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            # Fallback if radar chart fails
            return None
    
    def get_recommendations_text(self, valid_results) -> Dict:
        """Get AI-powered recommendations as structured text data"""
        return self._analyze_and_recommend(valid_results)
    
    def _analyze_and_recommend(self, valid_results) -> Dict:
        """Analyze metrics and generate concise AI-powered recommendations"""
        recommendations = {
            'overall_best': None,
            'use_cases': {},
            'insights': []
        }
        
        if not valid_results:
            return recommendations
        
        # Calculate overall scores (weighted average)
        algorithm_scores = {}
        for alg, metrics in valid_results.items():
            # Weighted score: Accuracy (40%), F1 (30%), Precision (15%), Recall (15%)
            overall_score = (
                metrics['accuracy'] * 0.4 +
                metrics['f1_score'] * 0.3 +
                metrics['precision'] * 0.15 +
                metrics['recall'] * 0.15
            )
            algorithm_scores[alg] = {
                'name': metrics['name'],
                'score': overall_score,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1_score'],
                'confidence': metrics['avg_confidence']
            }
        
        # Find overall best
        best_alg = max(algorithm_scores.items(), key=lambda x: x[1]['score'])
        recommendations['overall_best'] = best_alg[1]
        
        # Find best for specific use cases with concise explanations
        best_accuracy = max(algorithm_scores.items(), key=lambda x: x[1]['accuracy'])
        recommendations['use_cases']['Maximum Accuracy'] = {
            'name': best_accuracy[1]['name'],
            'reason': f"{best_accuracy[1]['accuracy']:.1f}% accuracy. Use when correctness is critical."
        }
        
        best_f1 = max(algorithm_scores.items(), key=lambda x: x[1]['f1'])
        recommendations['use_cases']['Balanced Performance'] = {
            'name': best_f1[1]['name'],
            'reason': f"{best_f1[1]['f1']:.1f}% F1 score. Best overall balance of precision and recall."
        }
        
        best_precision = max(algorithm_scores.items(), key=lambda x: x[1]['precision'])
        recommendations['use_cases']['Minimize False Positives'] = {
            'name': best_precision[1]['name'],
            'reason': f"{best_precision[1]['precision']:.1f}% precision. Use when false positives are costly (e.g., spam detection)."
        }
        
        best_recall = max(algorithm_scores.items(), key=lambda x: x[1]['recall'])
        recommendations['use_cases']['Minimize False Negatives'] = {
            'name': best_recall[1]['name'],
            'reason': f"{best_recall[1]['recall']:.1f}% recall. Use when missing positives is costly (e.g., medical diagnosis)."
        }
        
        best_confidence = max(algorithm_scores.items(), key=lambda x: x[1]['confidence'])
        recommendations['use_cases']['High Confidence'] = {
            'name': best_confidence[1]['name'],
            'reason': f"{best_confidence[1]['confidence']:.1f}% avg confidence. Most reliable for production systems."
        }
        
        # Generate concise insights
        accuracy_range = max(m['accuracy'] for m in algorithm_scores.values()) - min(m['accuracy'] for m in algorithm_scores.values())
        if accuracy_range > 10:
            recommendations['insights'].append(f"Performance varies by {accuracy_range:.1f}%. Algorithm choice significantly impacts results.")
        else:
            recommendations['insights'].append("Similar performance across algorithms. Consider speed and resource usage.")
        
        top_3 = sorted(algorithm_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
        if top_3[0][1]['score'] - top_3[2][1]['score'] < 5:
            top_names = ', '.join([t[1]['name'] for t in top_3])
            recommendations['insights'].append(f"Top performers ({top_names}) are closely matched.")
        
        precision_leader = best_precision[1]
        recall_leader = best_recall[1]
        if precision_leader['name'] != recall_leader['name']:
            recommendations['insights'].append(f"Precision-Recall trade-off: {precision_leader['name']} (precision) vs {recall_leader['name']} (recall).")
        
        return recommendations
    
    def _create_scatter_plot(self, valid_results) -> str:
        """Create scatter plot: Accuracy vs Confidence"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        accuracies = [v['accuracy'] for v in valid_results.values()]
        confidences = [v['avg_confidence'] for v in valid_results.values()]
        names = [v['name'] for v in valid_results.values()]
        
        scatter = ax.scatter(confidences, accuracies, s=200, alpha=0.6, c=range(len(accuracies)), cmap='viridis')
        
        for i, name in enumerate(names):
            ax.annotate(name, (confidences[i], accuracies[i]), fontsize=9, ha='center', va='bottom')
        
        ax.set_xlabel('Average Confidence (%)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy vs Confidence - Scatter Plot', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 105])
        ax.set_ylim([0, 105])
        
        plt.colorbar(scatter, ax=ax, label='Algorithm Index')
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_pie_chart(self, valid_results) -> str:
        """Create pie chart showing top performers"""
        # Calculate overall score (average of all metrics)
        scores = {}
        for alg, metrics in valid_results.items():
            overall = (metrics['accuracy'] + metrics['precision'] + metrics['recall'] + metrics['f1_score']) / 4
            scores[metrics['name']] = overall
        
        # Get top 5
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        labels = [x[0] for x in sorted_scores]
        values = [x[1] for x in sorted_scores]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax.set_title('Top 5 Algorithms by Overall Performance\n(Pie Chart)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_horizontal_bar_chart(self, algorithms, accuracies, precisions, recalls, f1_scores) -> str:
        """Create horizontal bar chart for ranking"""
        # Calculate overall score
        overall_scores = [(a + p + r + f) / 4 for a, p, r, f in zip(accuracies, precisions, recalls, f1_scores)]
        
        # Sort by score
        sorted_data = sorted(zip(algorithms, overall_scores, accuracies, precisions, recalls, f1_scores), 
                           key=lambda x: x[1], reverse=True)
        
        sorted_algs = [x[0] for x in sorted_data]
        sorted_scores = [x[1] for x in sorted_data]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(sorted_algs))
        bars = ax.barh(y_pos, sorted_scores, color=plt.cm.viridis(np.linspace(0, 1, len(sorted_algs))))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_algs)
        ax.set_xlabel('Overall Score (%)', fontsize=12)
        ax.set_title('Algorithm Ranking - Horizontal Bar Chart', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 105])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(sorted_scores):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        return img_base64
    
    def create_comparison_chart(self, results: Dict) -> str:
        """Legacy method for backward compatibility"""
        charts = self.create_all_charts(results)
        return charts.get('grouped_bar', None)
    
    def create_metrics_table(self, results: Dict) -> pd.DataFrame:
        """
        Create a metrics comparison table
        
        Args:
            results: Evaluation results
        
        Returns:
            DataFrame with comparison metrics
        """
        data = []
        for alg, metrics in results.items():
            if 'error' not in metrics:
                data.append({
                    'Algorithm': metrics['name'],
                    'Accuracy (%)': metrics['accuracy'],
                    'Precision (%)': metrics['precision'],
                    'Recall (%)': metrics['recall'],
                    'F1 Score (%)': metrics['f1_score'],
                    'Avg Confidence (%)': metrics['avg_confidence']
                })
        
        return pd.DataFrame(data)


