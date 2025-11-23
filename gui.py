#!/usr/bin/env python3
"""
Modern GUI Application for Multi-Algorithm Sentiment Analysis System
Built with CustomTkinter for a sleek, modern appearance
"""

import customtkinter as ctk
from tkinter import scrolledtext
import sys
from src.sentiment_analyzer import SentimentAnalyzer
from src.algorithm_comparison import AlgorithmComparator
from src.test_reviews import get_test_reviews, get_positive_reviews, get_negative_reviews, get_neutral_reviews, get_complex_reviews
import threading
from typing import List
from PIL import Image
import io
import base64

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SentimentAnalysisGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Multi-Algorithm Sentiment Analysis")
        self.root.geometry("1400x900")
        
        # Set window attributes
        self.root.attributes("-alpha", 0.97)
        
        # Initialize analyzer
        self.analyzer = None
        self.comparator = None
        self.init_analyzer()
        
        # Create UI
        self.setup_ui()
        
    def init_analyzer(self):
        """Initialize the sentiment analyzer in a separate thread"""
        def load():
            try:
                self.analyzer = SentimentAnalyzer()
                self.comparator = AlgorithmComparator(self.analyzer)
                print("Sentiment analyzer loaded successfully!")
            except Exception as e:
                print(f"Error loading analyzer: {e}")
                sys.exit(1)
        
        loading_thread = threading.Thread(target=load, daemon=True)
        loading_thread.start()
        loading_thread.join()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Main container
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Multi-Algorithm Sentiment Analysis",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title_label.pack()
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Compare and analyze sentiment using multiple algorithms",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle_label.pack()
        
        # Create tabbed interface
        self.tabview = ctk.CTkTabview(main_container, height=700)
        self.tabview.pack(fill="both", expand=True)
        
        # Add tabs
        self.tabview.add("Analyze Sentiment")
        self.tabview.add("Compare Algorithms")
        self.tabview.add("Test Reviews")
        self.tabview.add("Performance Metrics")
        
        # Setup each tab
        self.setup_analyze_tab()
        self.setup_compare_tab()
        self.setup_test_reviews_tab()
        self.setup_metrics_tab()
        
        # Footer
        footer = ctk.CTkLabel(
            main_container,
            text="FDS Capstone Project • Multi-Algorithm Sentiment Analysis",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        footer.pack(pady=(10, 0))
    
    def setup_analyze_tab(self):
        """Setup the analyze sentiment tab"""
        tab = self.tabview.tab("Analyze Sentiment")
        
        # Left side - Input
        left_frame = ctk.CTkFrame(tab, corner_radius=15)
        left_frame.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=20)
        
        input_label = ctk.CTkLabel(
            left_frame,
            text="Enter Text to Analyze",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        input_label.pack(pady=(15, 10))
        
        # Algorithm selection
        alg_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        alg_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(alg_frame, text="Select Algorithm:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        
        # Get available algorithms
        available_algs = self.analyzer.get_available_algorithms()
        alg_names = []
        self.algorithm_map = {}
        
        for alg in available_algs:
            alg_info = self.analyzer.algorithms[alg]
            name = alg_info['name']
            alg_names.append(name)
            self.algorithm_map[name] = alg
        
        self.algorithm_var = ctk.StringVar(value=alg_names[0] if alg_names else "distilbert")
        self.algorithm_menu = ctk.CTkOptionMenu(
            alg_frame,
            values=alg_names,
            variable=self.algorithm_var,
            width=250
        )
        self.algorithm_menu.pack(side="left")
        
        # Text input
        self.review_textbox = ctk.CTkTextbox(
            left_frame,
            font=ctk.CTkFont(size=13),
            corner_radius=10,
            height=300
        )
        self.review_textbox.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        self.review_textbox.insert("1.0", "Enter your text here...")
        
        analyze_btn = ctk.CTkButton(
            left_frame,
            text="Analyze Sentiment",
            command=self.analyze_sentiment,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10
        )
        analyze_btn.pack(pady=(0, 15))
        
        # Right side - Results
        right_frame = ctk.CTkFrame(tab, corner_radius=15)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        results_label = ctk.CTkLabel(
            right_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_label.pack(pady=(15, 10))
        
        self.sentiment_result = ctk.CTkLabel(
            right_frame,
            text="No analysis yet",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="gray"
        )
        self.sentiment_result.pack(pady=20)
        
        self.algorithm_label = ctk.CTkLabel(
            right_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.algorithm_label.pack()
        
        self.confidence_label = ctk.CTkLabel(
            right_frame,
            text="",
            font=ctk.CTkFont(size=16)
        )
        self.confidence_label.pack(pady=(10, 5))
        
        self.confidence_bar = ctk.CTkProgressBar(
            right_frame,
            corner_radius=10,
            height=25,
            width=400
        )
        self.confidence_bar.pack(pady=10)
        self.confidence_bar.set(0)
    
    def setup_compare_tab(self):
        """Setup the compare algorithms tab"""
        tab = self.tabview.tab("Compare Algorithms")
        
        # Input section
        input_frame = ctk.CTkFrame(tab, corner_radius=15)
        input_frame.pack(fill="x", padx=20, pady=20)
        
        input_label = ctk.CTkLabel(
            input_frame,
            text="Enter Text to Compare",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        input_label.pack(pady=(15, 10))
        
        self.compare_textbox = ctk.CTkTextbox(
            input_frame,
            font=ctk.CTkFont(size=13),
            corner_radius=10,
            height=150
        )
        self.compare_textbox.pack(fill="x", padx=20, pady=(0, 20))
        self.compare_textbox.insert("1.0", "Enter your text here...")
        
        compare_btn = ctk.CTkButton(
            input_frame,
            text="Compare All Algorithms",
            command=self.compare_algorithms,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10
        )
        compare_btn.pack(pady=(0, 15))
        
        # Results section
        results_frame = ctk.CTkFrame(tab, corner_radius=15)
        results_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        results_label = ctk.CTkLabel(
            results_frame,
            text="Comparison Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_label.pack(pady=(15, 10))
        
        # Scrollable frame for results
        self.compare_scroll = ctk.CTkScrollableFrame(
            results_frame,
            corner_radius=10,
            fg_color="transparent"
        )
        self.compare_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        self.compare_placeholder = ctk.CTkLabel(
            self.compare_scroll,
            text="Enter text and click 'Compare All Algorithms' to see results",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.compare_placeholder.pack(pady=50)
    
    def setup_test_reviews_tab(self):
        """Setup the test reviews tab"""
        tab = self.tabview.tab("Test Reviews")
        
        # Header
        header_frame = ctk.CTkFrame(tab, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Comprehensive Test Dataset",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack()
        
        desc_label = ctk.CTkLabel(
            header_frame,
            text="150 reviews covering positive, negative, neutral, double negations, sarcasm, and edge cases",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc_label.pack(pady=(5, 15))
        
        # Filter buttons
        filter_frame = ctk.CTkFrame(tab, fg_color="transparent")
        filter_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(filter_frame, text="Filter by category:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        
        self.review_filter = ctk.StringVar(value="All")
        filter_options = ["All", "Positive", "Negative", "Neutral", "Double Negations", "Sarcasm"]
        
        for option in filter_options:
            ctk.CTkRadioButton(
                filter_frame,
                text=option,
                variable=self.review_filter,
                value=option,
                command=self.filter_reviews
            ).pack(side="left", padx=5)
        
        # Stats frame
        stats_frame = ctk.CTkFrame(tab, corner_radius=10)
        stats_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        all_reviews = get_test_reviews()
        positive_count = len(get_positive_reviews())
        negative_count = len(get_negative_reviews())
        neutral_count = len(get_neutral_reviews())
        complex_count = len(get_complex_reviews())
        
        stats_text = f"Total: {len(all_reviews)} reviews | Positive: {positive_count} | Negative: {negative_count} | Neutral: {neutral_count} | Complex Cases: {complex_count}"
        ctk.CTkLabel(
            stats_frame,
            text=stats_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        ).pack(pady=10)
        
        # Scrollable frame for reviews
        self.reviews_scroll = ctk.CTkScrollableFrame(tab, corner_radius=15)
        self.reviews_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        self.display_reviews(all_reviews)
    
    def filter_reviews(self):
        """Filter reviews based on selected category"""
        filter_value = self.review_filter.get()
        
        if filter_value == "All":
            reviews = get_test_reviews()
        elif filter_value == "Positive":
            reviews = get_positive_reviews()
        elif filter_value == "Negative":
            reviews = get_negative_reviews()
        elif filter_value == "Neutral":
            reviews = get_neutral_reviews()
        elif filter_value == "Double Negations":
            reviews = get_complex_reviews()
        elif filter_value == "Sarcasm":
            # Sarcasm reviews are in positions 85-95
            all_reviews = get_test_reviews()
            reviews = all_reviews[85:95]
        else:
            reviews = get_test_reviews()
        
        # Clear and redisplay
        for widget in self.reviews_scroll.winfo_children():
            widget.destroy()
        
        self.display_reviews(reviews)
    
    def display_reviews(self, reviews):
        """Display reviews in the scrollable frame"""
        for idx, (text, label) in enumerate(reviews):
            review_card = ctk.CTkFrame(self.reviews_scroll, corner_radius=10)
            review_card.pack(fill="x", padx=10, pady=5)
            
            # Label badge
            label_frame = ctk.CTkFrame(review_card, fg_color="transparent")
            label_frame.pack(fill="x", padx=15, pady=(10, 5))
            
            if label == "Positive":
                label_color = "#28a745"
                bg_color = "#1e5a2e"
            elif label == "Negative":
                label_color = "#dc3545"
                bg_color = "#7a1e1e"
            else:
                label_color = "#ffc107"
                bg_color = "#7a5a1e"
            
            label_badge = ctk.CTkLabel(
                label_frame,
                text=f"Label: {label}",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=label_color,
                width=100,
                corner_radius=5,
                fg_color=bg_color
            )
            label_badge.pack(side="left")
            
            review_num = ctk.CTkLabel(
                label_frame,
                text=f"Review #{idx + 1}",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            review_num.pack(side="right")
            
            # Review text
            review_text = ctk.CTkLabel(
                review_card,
                text=text,
                font=ctk.CTkFont(size=12),
                wraplength=900,
                justify="left",
                anchor="w"
            )
            review_text.pack(fill="x", padx=15, pady=(0, 10))
            
            # Use button
            use_btn = ctk.CTkButton(
                review_card,
                text="Use This Review",
                command=lambda t=text: self.use_review(t),
                height=30,
                font=ctk.CTkFont(size=11),
                width=120
            )
            use_btn.pack(side="right", padx=15, pady=(0, 10))
    
    def use_review(self, review_text):
        """Use a review from the test dataset in the compare tab"""
        # Switch to compare tab
        self.tabview.set("Compare Algorithms")
        
        # Clear and insert text in compare textbox
        self.compare_textbox.delete("1.0", "end")
        self.compare_textbox.insert("1.0", review_text)
        
        # Automatically trigger comparison
        self.compare_algorithms()
    
    def setup_metrics_tab(self):
        """Setup the performance metrics tab"""
        tab = self.tabview.tab("Performance Metrics")
        
        # Header
        header_frame = ctk.CTkFrame(tab, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Algorithm Performance Metrics",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack()
        
        desc_label = ctk.CTkLabel(
            header_frame,
            text="View accuracy, precision, recall, F1 score, and visual comparisons in a new window.\nMetrics are calculated using 150 comprehensive test reviews including double negations, sarcasm, and edge cases.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            justify="center"
        )
        desc_label.pack(pady=(5, 15))
        
        # Button
        button_frame = ctk.CTkFrame(tab, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.metrics_button = ctk.CTkButton(
            button_frame,
            text="Open Performance Metrics Window",
            command=self.load_metrics,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10,
            width=300
        )
        self.metrics_button.pack(pady=20)
        
        # Info label
        info_label = ctk.CTkLabel(
            tab,
            text="Click the button above to open a new window with detailed performance metrics and visualizations",
            font=ctk.CTkFont(size=13),
            text_color="gray",
            wraplength=600
        )
        info_label.pack(pady=20)
    
    def analyze_sentiment(self):
        """Analyze sentiment with selected algorithm"""
        text = self.review_textbox.get("1.0", "end-1c").strip()
        
        if not text or text == "Enter your text here...":
            self.show_message("Please enter text to analyze", "error")
            return
        
        # Get selected algorithm
        alg_name = self.algorithm_var.get()
        algorithm = self.algorithm_map.get(alg_name, "distilbert")
        
        # Analyze
        sentiment, confidence, metadata = self.analyzer.analyze(text, algorithm)
        
        # Update UI
        self.sentiment_result.configure(
            text=sentiment,
            text_color=("#28a745" if sentiment == "Positive" else ("#dc3545" if sentiment == "Negative" else "#ffc107"))
        )
        self.algorithm_label.configure(text=f"Algorithm: {alg_name}")
        self.confidence_label.configure(text=f"Confidence: {confidence * 100:.2f}%")
        self.confidence_bar.set(confidence)
    
    def compare_algorithms(self):
        """Compare all algorithms on the same text"""
        text = self.compare_textbox.get("1.0", "end-1c").strip()
        
        if not text or text == "Enter your text here...":
            self.show_message("Please enter text to compare", "error")
            return
        
        # Clear previous results
        for widget in self.compare_scroll.winfo_children():
            widget.destroy()
        
        # Show loading
        loading_label = ctk.CTkLabel(
            self.compare_scroll,
            text="Comparing algorithms...",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        loading_label.pack(pady=50)
        self.root.update()
        
        # Compare
        results = self.analyzer.compare_algorithms(text)
        
        # Clear loading
        loading_label.destroy()
        
        # Display results
        for alg, result in results.items():
            card = ctk.CTkFrame(self.compare_scroll, corner_radius=10)
            card.pack(fill="x", padx=10, pady=5)
            
            # Algorithm name
            name_label = ctk.CTkLabel(
                card,
                text=result['name'],
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=("#1f6aa5", "#4a9eff")
            )
            name_label.pack(pady=(10, 5))
            
            # Sentiment
            sentiment_color = "#28a745" if result['sentiment'] == "Positive" else ("#dc3545" if result['sentiment'] == "Negative" else "#ffc107")
            sentiment_label = ctk.CTkLabel(
                card,
                text=result['sentiment'],
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=sentiment_color
            )
            sentiment_label.pack()
            
            # Confidence
            conf_label = ctk.CTkLabel(
                card,
                text=f"Confidence: {result['confidence'] * 100:.2f}%",
                font=ctk.CTkFont(size=14)
            )
            conf_label.pack(pady=(5, 10))
            
            # Progress bar
            progress = ctk.CTkProgressBar(card, width=300, height=15)
            progress.pack(pady=(0, 10))
            progress.set(result['confidence'])
    
    def load_metrics(self):
        """Load and display performance metrics in a new window"""
        self.metrics_button.configure(state="disabled", text="Evaluating...")
        
        def evaluate():
            try:
                results = self.comparator.evaluate_all_algorithms()
                charts = self.comparator.create_all_charts(results)
                self.root.after(0, lambda: self.open_metrics_window(results, charts))
            except Exception as e:
                self.root.after(0, lambda: self.display_metrics_error(str(e)))
        
        threading.Thread(target=evaluate, daemon=True).start()
    
    def open_metrics_window(self, results, charts):
        """Open a new window to display metrics"""
        self.metrics_button.configure(state="normal", text="Open Performance Metrics Window")
        
        # Create new window
        metrics_window = ctk.CTkToplevel(self.root)
        metrics_window.title("Performance Metrics")
        metrics_window.geometry("1400x900")
        
        # Header
        header_frame = ctk.CTkFrame(metrics_window, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Algorithm Performance Metrics",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title_label.pack()
        
        dataset_label = ctk.CTkLabel(
            header_frame,
            text="Evaluated on 150 test reviews: 50 Positive, 50 Negative, 35 Neutral, 15 Complex (double negations, sarcasm, edge cases)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        dataset_label.pack(pady=(5, 10))
        
        # Scrollable frame for charts and metrics
        scroll_frame = ctk.CTkScrollableFrame(metrics_window, corner_radius=15)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Chart titles mapping (without emojis)
        chart_titles = {
            'grouped_bar': 'Grouped Bar Chart - All Metrics Comparison',
            'line': 'Line Chart - Metrics Trend',
            'radar': 'Radar Chart - Multi-Metric Comparison',
            'recommendations': 'AI Recommendations - Best Algorithm Selection Guide',
            'scatter': 'Scatter Plot - Accuracy vs Confidence',
            'pie': 'Pie Chart - Top 5 Algorithms',
            'horizontal_bar': 'Horizontal Bar Chart - Algorithm Ranking'
        }
        
        # Display all charts (excluding recommendations which will be text)
        chart_order = ['grouped_bar', 'line', 'horizontal_bar', 'radar', 'scatter', 'pie']
        
        for chart_type in chart_order:
            if chart_type in charts and charts[chart_type]:
                self._display_chart_in_window(scroll_frame, chart_type, charts[chart_type], chart_titles.get(chart_type, chart_type))
        
        # Display AI recommendations as text
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            recommendations = self.comparator.get_recommendations_text(valid_results)
            self._display_recommendations_text(scroll_frame, recommendations)
        
        # Display metrics summary table
        self._display_metrics_table_in_window(scroll_frame, results)
    
    def _display_chart_in_window(self, parent, chart_type, chart_base64, title):
        """Display a single chart in the metrics window"""
        chart_frame = ctk.CTkFrame(parent, corner_radius=10)
        chart_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            chart_frame,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title_label.pack(pady=(15, 10))
        
        # Convert base64 to image
        try:
            img_data = base64.b64decode(chart_base64)
            img = Image.open(io.BytesIO(img_data))
            
            # Resize if too large (max width 1200px)
            max_width = 1200
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to CTkImage
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            
            # Display image
            img_label = ctk.CTkLabel(chart_frame, image=ctk_img, text="")
            img_label.pack(pady=(0, 15))
        except Exception as e:
            error_label = ctk.CTkLabel(
                chart_frame,
                text=f"Error displaying chart: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="red"
            )
            error_label.pack(pady=10)
    
    def _display_recommendations_text(self, parent, recommendations):
        """Display AI recommendations as text in the metrics window"""
        rec_frame = ctk.CTkFrame(parent, corner_radius=10)
        rec_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            rec_frame,
            text="AI Recommendations - Algorithm Selection Guide",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title_label.pack(pady=(15, 10))
        
        # Overall Best
        if recommendations['overall_best']:
            best = recommendations['overall_best']
            best_frame = ctk.CTkFrame(rec_frame, fg_color="transparent")
            best_frame.pack(fill="x", padx=20, pady=5)
            
            ctk.CTkLabel(
                best_frame,
                text="Overall Best Performer:",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#28a745"
            ).pack(anchor="w")
            
            metrics_text = f"{best['name']} - Accuracy: {best['accuracy']:.1f}% | F1: {best['f1']:.1f}% | Precision: {best['precision']:.1f}% | Recall: {best['recall']:.1f}%"
            ctk.CTkLabel(
                best_frame,
                text=metrics_text,
                font=ctk.CTkFont(size=12),
                text_color="gray"
            ).pack(anchor="w", pady=(5, 10))
        
        # Use Cases
        use_case_frame = ctk.CTkFrame(rec_frame, fg_color="transparent")
        use_case_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            use_case_frame,
            text="Recommendations by Use Case:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        ).pack(anchor="w", pady=(0, 10))
        
        for use_case, alg_info in recommendations['use_cases'].items():
            case_frame = ctk.CTkFrame(use_case_frame, fg_color="transparent")
            case_frame.pack(fill="x", pady=3)
            
            ctk.CTkLabel(
                case_frame,
                text=f"• {use_case}:",
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(anchor="w")
            
            ctk.CTkLabel(
                case_frame,
                text=f"  {alg_info['name']} - {alg_info['reason']}",
                font=ctk.CTkFont(size=11),
                text_color="gray",
                wraplength=1000
            ).pack(anchor="w", padx=(15, 0))
        
        # Insights
        if recommendations['insights']:
            insights_frame = ctk.CTkFrame(rec_frame, fg_color="transparent")
            insights_frame.pack(fill="x", padx=20, pady=(15, 10))
            
            ctk.CTkLabel(
                insights_frame,
                text="Key Insights:",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#ff9800"
            ).pack(anchor="w", pady=(0, 10))
            
            for insight in recommendations['insights']:
                ctk.CTkLabel(
                    insights_frame,
                    text=f"• {insight}",
                    font=ctk.CTkFont(size=11),
                    text_color="gray",
                    wraplength=1000
                ).pack(anchor="w", pady=2)
    
    def _display_metrics_table_in_window(self, parent, results):
        """Display metrics summary table in the metrics window"""
        table_frame = ctk.CTkFrame(parent, corner_radius=10)
        table_frame.pack(fill="x", padx=10, pady=10)
        
        title_label = ctk.CTkLabel(
            table_frame,
            text="Detailed Metrics Summary",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title_label.pack(pady=(15, 10))
        
        # Header row
        header_frame = ctk.CTkFrame(table_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(0, 5))
        
        headers = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Avg Confidence"]
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=12, weight="bold"),
                width=120 if i == 0 else 100
            )
            label.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
            header_frame.grid_columnconfigure(i, weight=1)
        
        # Data rows
        for idx, (alg, metrics) in enumerate(results.items()):
            if 'error' in metrics:
                continue
            
            row_frame = ctk.CTkFrame(table_frame, fg_color="transparent" if idx % 2 == 0 else ("#2b2b2b", "#1a1a1a"))
            row_frame.pack(fill="x", padx=20, pady=2)
            
            values = [
                metrics['name'],
                f"{metrics['accuracy']:.2f}%",
                f"{metrics['precision']:.2f}%",
                f"{metrics['recall']:.2f}%",
                f"{metrics['f1_score']:.2f}%",
                f"{metrics['avg_confidence']:.2f}%"
            ]
            
            for i, value in enumerate(values):
                label = ctk.CTkLabel(
                    row_frame,
                    text=value,
                    font=ctk.CTkFont(size=11),
                    width=120 if i == 0 else 100
                )
                label.grid(row=0, column=i, padx=5, pady=3, sticky="ew")
                row_frame.grid_columnconfigure(i, weight=1)
    
    def display_metrics_error(self, error_msg):
        """Display error message"""
        self.metrics_button.configure(state="normal", text="Open Performance Metrics Window")
        
        for widget in self.metrics_scroll.winfo_children():
            widget.destroy()
        
        error_label = ctk.CTkLabel(
            self.metrics_scroll,
            text=f"❌ Error: {error_msg}",
            font=ctk.CTkFont(size=13),
            text_color="red"
        )
        error_label.pack(pady=50)
    
    def show_message(self, message, msg_type="info"):
        """Show a message to the user"""
        # Simple message display - can be enhanced
        print(f"{msg_type.upper()}: {message}")


def main():
    """Main entry point"""
    try:
        app = SentimentAnalysisGUI()
        app.root.mainloop()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
