#!/usr/bin/env python3
"""
Modern GUI Application for Movie Recommendation System with Sentiment Analysis
Built with CustomTkinter for a sleek, modern appearance
"""

import customtkinter as ctk
from tkinter import scrolledtext
import sys
from src.movie_recommender import MovieRecommender
from src.evaluator import ModelEvaluator
import threading
from typing import List

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"


class ExpandableMovieCard(ctk.CTkFrame):
    """Interactive expandable card for movie recommendations"""
    
    def __init__(self, parent, movie_data, index, gui_instance=None, **kwargs):
        super().__init__(parent, corner_radius=12, **kwargs)
        
        self.movie_data = movie_data
        self.index = index
        self.gui_instance = gui_instance  # Store GUI instance
        self.is_expanded = False
        
        # Configure colors
        self.configure(fg_color=("#e0e0e0", "#2b2b2b"), border_width=2, border_color=("#c0c0c0", "#404040"))
        
        # Main container
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the card UI"""
        # Header (always visible) - clickable
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent", cursor="hand2")
        self.header_frame.pack(fill="x", padx=10, pady=8)
        
        # Bind click to toggle
        self.header_frame.bind("<Button-1>", lambda e: self.toggle_expand())
        
        # Title row
        title_row = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_row.pack(fill="x")
        
        # Index number
        index_label = ctk.CTkLabel(
            title_row,
            text=f"{self.index}.",
            font=ctk.CTkFont(size=16, weight="bold"),
            width=30,
            text_color=("#1f6aa5", "#4a9eff")
        )
        index_label.pack(side="left")
        index_label.bind("<Button-1>", lambda e: self.toggle_expand())
        
        # Movie title
        self.title_label = ctk.CTkLabel(
            title_row,
            text=self.movie_data['title'],
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w"
        )
        self.title_label.pack(side="left", fill="x", expand=True, padx=5)
        self.title_label.bind("<Button-1>", lambda e: self.toggle_expand())
        
        # Expand/collapse icon
        self.expand_icon = ctk.CTkLabel(
            title_row,
            text="▼",
            font=ctk.CTkFont(size=12),
            width=20,
            text_color="gray"
        )
        self.expand_icon.pack(side="right")
        self.expand_icon.bind("<Button-1>", lambda e: self.toggle_expand())
        
        # Similarity score (always visible)
        similarity_label = ctk.CTkLabel(
            self.header_frame,
            text=f"Match Score: {self.movie_data['similarity']:.2%}",
            font=ctk.CTkFont(size=12),
            text_color=("#2e7d32", "#66bb6a"),
            anchor="w"
        )
        similarity_label.pack(fill="x", pady=(3, 0))
        similarity_label.bind("<Button-1>", lambda e: self.toggle_expand())
        
        # Details frame (expandable)
        self.details_frame = ctk.CTkFrame(self, fg_color="transparent")
        # Don't pack yet - will be shown on expand
        
        # Populate details
        self.setup_details()
    
    def setup_details(self):
        """Setup the expandable details section"""
        # Add some padding
        details_container = ctk.CTkFrame(self.details_frame, fg_color="transparent")
        details_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        # Genres
        if 'genres' in self.movie_data and self.movie_data['genres']:
            genres_frame = ctk.CTkFrame(details_container, fg_color="transparent")
            genres_frame.pack(fill="x", pady=3)
            
            ctk.CTkLabel(
                genres_frame,
                text="🎭 Genres:",
                font=ctk.CTkFont(size=12, weight="bold"),
                width=100,
                anchor="w"
            ).pack(side="left")
            
            ctk.CTkLabel(
                genres_frame,
                text=self.movie_data['genres'],
                font=ctk.CTkFont(size=12),
                anchor="w",
                wraplength=400
            ).pack(side="left", fill="x", expand=True)
        
        # Director
        if 'director' in self.movie_data and self.movie_data['director']:
            director_frame = ctk.CTkFrame(details_container, fg_color="transparent")
            director_frame.pack(fill="x", pady=3)
            
            ctk.CTkLabel(
                director_frame,
                text="🎬 Director:",
                font=ctk.CTkFont(size=12, weight="bold"),
                width=100,
                anchor="w"
            ).pack(side="left")
            
            ctk.CTkLabel(
                director_frame,
                text=self.movie_data['director'],
                font=ctk.CTkFont(size=12),
                anchor="w"
            ).pack(side="left", fill="x", expand=True)
        
        # Cast
        if 'actors' in self.movie_data and self.movie_data['actors']:
            actors_text = ", ".join([a for a in self.movie_data['actors'] if a and str(a) != 'nan'])
            if actors_text:
                cast_frame = ctk.CTkFrame(details_container, fg_color="transparent")
                cast_frame.pack(fill="x", pady=3)
                
                ctk.CTkLabel(
                    cast_frame,
                    text="⭐ Cast:",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    width=100,
                    anchor="w"
                ).pack(side="left", anchor="n")
                
                ctk.CTkLabel(
                    cast_frame,
                    text=actors_text,
                    font=ctk.CTkFont(size=12),
                    anchor="w",
                    wraplength=400,
                    justify="left"
                ).pack(side="left", fill="x", expand=True)
        
        # Rating if available
        if 'vote_average' in self.movie_data and self.movie_data['vote_average']:
            rating_frame = ctk.CTkFrame(details_container, fg_color="transparent")
            rating_frame.pack(fill="x", pady=3)
            
            ctk.CTkLabel(
                rating_frame,
                text="⭐ Rating:",
                font=ctk.CTkFont(size=12, weight="bold"),
                width=100,
                anchor="w"
            ).pack(side="left")
            
            ctk.CTkLabel(
                rating_frame,
                text=f"{self.movie_data['vote_average']}/10",
                font=ctk.CTkFont(size=12),
                anchor="w"
            ).pack(side="left")
        
        # Action buttons
        actions_frame = ctk.CTkFrame(details_container, fg_color="transparent")
        actions_frame.pack(fill="x", pady=(10, 0))
        
        # Get more info button
        ctk.CTkButton(
            actions_frame,
            text="Full Details",
            command=lambda: self.show_full_details(),
            height=28,
            font=ctk.CTkFont(size=11),
            corner_radius=8,
            width=120
        ).pack(side="left", padx=2)
        
        # Find similar button
        ctk.CTkButton(
            actions_frame,
            text="Similar Movies",
            command=lambda: self.find_similar(),
            height=28,
            font=ctk.CTkFont(size=11),
            corner_radius=8,
            width=130,
            fg_color=("#2e7d32", "#1b5e20")
        ).pack(side="left", padx=2)
    
    def toggle_expand(self):
        """Toggle card expansion"""
        self.is_expanded = not self.is_expanded
        
        if self.is_expanded:
            # Expand
            self.details_frame.pack(fill="x", padx=5, pady=(0, 5))
            self.expand_icon.configure(text="▲")
            self.configure(border_color=("#1f6aa5", "#4a9eff"))
        else:
            # Collapse
            self.details_frame.pack_forget()
            self.expand_icon.configure(text="▼")
            self.configure(border_color=("#c0c0c0", "#404040"))
    
    def show_full_details(self):
        """Show full movie details in info tab"""
        if self.gui_instance:
            self.gui_instance.tabview.set("ℹ️ Movie Info")
            self.gui_instance.info_entry.delete(0, "end")
            self.gui_instance.info_entry.insert(0, self.movie_data['title'])
            self.gui_instance.get_movie_info()
    
    def find_similar(self):
        """Find similar movies to this one"""
        if self.gui_instance:
            self.gui_instance.search_entry.delete(0, "end")
            self.gui_instance.search_entry.insert(0, self.movie_data['title'])
            self.gui_instance.search_movies()


class MovieRecommenderGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🎬 Movie Recommender & Sentiment Analyzer")
        self.root.geometry("1200x800")
        
        # Set window attributes for transparency and modern look
        self.root.attributes("-alpha", 0.97)  # Slight transparency
        
        # Initialize recommender system
        self.recommender = None
        self.init_recommender()
        
        # Autocomplete variables
        self.autocomplete_window = None
        self.autocomplete_listbox = None
        self.search_suggestions = []
        self.info_suggestions = []
        
        # Recommendations storage
        self.recommendation_widgets = []
        self.current_recommendations = []
        
        # Create UI
        self.setup_ui()
        
    def init_recommender(self):
        """Initialize the movie recommender system in a separate thread"""
        def load():
            try:
                self.recommender = MovieRecommender()
                print("✓ Recommender system loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading recommender: {e}")
                sys.exit(1)
        
        loading_thread = threading.Thread(target=load, daemon=True)
        loading_thread.start()
        loading_thread.join()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Main container with padding
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎬 Movie Recommender & Sentiment Analyzer",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title_label.pack()
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Discover movies you'll love • Analyze reviews with AI",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle_label.pack()
        
        # Create tabbed interface
        self.tabview = ctk.CTkTabview(main_container, height=600)
        self.tabview.pack(fill="both", expand=True)
        
        # Add tabs
        self.tabview.add("🔍 Search & Recommend")
        self.tabview.add("💭 Sentiment Analysis")
        self.tabview.add("ℹ️ Movie Info")
        self.tabview.add("📊 Model Metrics")
        
        # Setup each tab
        self.setup_recommend_tab()
        self.setup_sentiment_tab()
        self.setup_info_tab()
        self.setup_metrics_tab()
        
        # Footer
        footer = ctk.CTkLabel(
            main_container,
            text="FDS Capstone Project • Powered by ML & NLP",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        footer.pack(pady=(10, 0))
    
    def setup_recommend_tab(self):
        """Setup the recommendation tab"""
        tab = self.tabview.tab("🔍 Search & Recommend")
        
        # Search section
        search_frame = ctk.CTkFrame(tab, corner_radius=15)
        search_frame.pack(fill="x", padx=20, pady=20)
        
        search_label = ctk.CTkLabel(
            search_frame,
            text="Search for a Movie",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        search_label.pack(pady=(15, 10))
        
        # Search input
        input_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        self.search_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type movie name (e.g., The Matrix, Inception)...",
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=10
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.search_entry.bind("<KeyRelease>", lambda e: self.on_search_key_release(self.search_entry, "search"))
        
        search_btn = ctk.CTkButton(
            input_frame,
            text="Search",
            command=self.search_movies,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            width=120
        )
        search_btn.pack(side="right")
        
        # Number of recommendations slider
        slider_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        slider_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        slider_label = ctk.CTkLabel(
            slider_frame,
            text="Number of Recommendations:",
            font=ctk.CTkFont(size=13)
        )
        slider_label.pack(side="left")
        
        self.num_recommendations = ctk.IntVar(value=10)
        self.rec_slider = ctk.CTkSlider(
            slider_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            variable=self.num_recommendations,
            width=200
        )
        self.rec_slider.pack(side="left", padx=10)
        
        self.rec_count_label = ctk.CTkLabel(
            slider_frame,
            text="10",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=30
        )
        self.rec_count_label.pack(side="left")
        
        self.rec_slider.configure(command=lambda v: self.rec_count_label.configure(text=str(int(v))))
        
        # Results section - Scrollable frame for interactive cards
        results_frame = ctk.CTkFrame(tab, corner_radius=15)
        results_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        results_label = ctk.CTkLabel(
            results_frame,
            text="Recommendations",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_label.pack(pady=(15, 10))
        
        # Scrollable frame for movie cards
        self.results_scrollable_frame = ctk.CTkScrollableFrame(
            results_frame,
            corner_radius=10,
            fg_color="transparent"
        )
        self.results_scrollable_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        # Initial placeholder message
        self.results_placeholder = ctk.CTkLabel(
            self.results_scrollable_frame,
            text="Search for a movie to get personalized recommendations...\n\n"
                 "Try movies like:\n"
                 "• The Dark Knight\n"
                 "• Inception\n"
                 "• The Matrix\n"
                 "• Pulp Fiction\n"
                 "• Interstellar",
            font=ctk.CTkFont(size=13),
            text_color="gray",
            justify="left"
        )
        self.results_placeholder.pack(pady=20)
    
    def setup_sentiment_tab(self):
        """Setup the sentiment analysis tab"""
        tab = self.tabview.tab("💭 Sentiment Analysis")
        
        # Input section
        input_frame = ctk.CTkFrame(tab, corner_radius=15)
        input_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        input_label = ctk.CTkLabel(
            input_frame,
            text="Write Your Movie Review",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        input_label.pack(pady=(15, 10))
        
        self.review_textbox = ctk.CTkTextbox(
            input_frame,
            font=ctk.CTkFont(size=13),
            corner_radius=10,
            height=200,
            wrap="word"
        )
        self.review_textbox.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        self.review_textbox.insert("1.0", "Enter your movie review here...")
        
        analyze_btn = ctk.CTkButton(
            input_frame,
            text="🔍 Analyze Sentiment",
            command=self.analyze_sentiment,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10
        )
        analyze_btn.pack(pady=(0, 15))
        
        # Results section
        results_frame = ctk.CTkFrame(tab, corner_radius=15)
        results_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        results_label = ctk.CTkLabel(
            results_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_label.pack(pady=(15, 10))
        
        self.sentiment_result = ctk.CTkLabel(
            results_frame,
            text="No analysis yet",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="gray"
        )
        self.sentiment_result.pack(pady=10)
        
        self.confidence_label = ctk.CTkLabel(
            results_frame,
            text="",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_label.pack(pady=(0, 15))
        
        # Progress bar for confidence
        self.confidence_bar = ctk.CTkProgressBar(
            results_frame,
            corner_radius=10,
            height=20,
            width=400
        )
        self.confidence_bar.pack(pady=(0, 15))
        self.confidence_bar.set(0)
    
    def setup_info_tab(self):
        """Setup the movie info tab"""
        tab = self.tabview.tab("ℹ️ Movie Info")
        
        # Search section
        search_frame = ctk.CTkFrame(tab, corner_radius=15)
        search_frame.pack(fill="x", padx=20, pady=20)
        
        search_label = ctk.CTkLabel(
            search_frame,
            text="Get Movie Details",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        search_label.pack(pady=(15, 10))
        
        input_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        self.info_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter exact movie title...",
            height=40,
            font=ctk.CTkFont(size=14),
            corner_radius=10
        )
        self.info_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.info_entry.bind("<KeyRelease>", lambda e: self.on_search_key_release(self.info_entry, "info"))
        
        info_btn = ctk.CTkButton(
            input_frame,
            text="Get Info",
            command=self.get_movie_info,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            width=120
        )
        info_btn.pack(side="right")
        
        # Info display
        info_frame = ctk.CTkFrame(tab, corner_radius=15)
        info_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        self.info_textbox = ctk.CTkTextbox(
            info_frame,
            font=ctk.CTkFont(size=13),
            corner_radius=10,
            wrap="word"
        )
        self.info_textbox.pack(fill="both", expand=True, padx=20, pady=20)
        self.info_textbox.insert("1.0", "Enter a movie title to see detailed information...")
        self.info_textbox.configure(state="disabled")
    
    def search_movies(self):
        """Search for movies and get recommendations"""
        movie_title = self.search_entry.get().strip()
        num_recs = int(self.num_recommendations.get())
        
        if not movie_title:
            self.show_message("Please enter a movie title", "error")
            return
        
        # Clear previous results and widgets
        self.clear_recommendation_widgets()
        
        # Show loading placeholder
        if self.results_placeholder:
            self.results_placeholder.destroy()
        
        loading_label = ctk.CTkLabel(
            self.results_scrollable_frame,
            text="🔍 Searching for recommendations...",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        loading_label.pack(pady=20)
        self.root.update()
        
        # Get recommendations
        recommendations, error = self.recommender.recommend_movies(movie_title, num_recs)
        
        # Remove loading label
        loading_label.destroy()
        
        if error:
            # Show error message
            error_label = ctk.CTkLabel(
                self.results_scrollable_frame,
                text=f"❌ Error: {error}\n\nTry searching for a different movie or check the spelling.",
                font=ctk.CTkFont(size=13),
                text_color=("#c62828", "#ef5350"),
                justify="left"
            )
            error_label.pack(pady=20)
        else:
            # Show header
            header_label = ctk.CTkLabel(
                self.results_scrollable_frame,
                text=f"✨ {len(recommendations)} Recommendations based on: {movie_title}",
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=("#1f6aa5", "#4a9eff"),
                anchor="w"
            )
            header_label.pack(fill="x", pady=(5, 15), padx=10)
            
            # Tip label
            tip_label = ctk.CTkLabel(
                self.results_scrollable_frame,
                text="💡 Click on any card to expand and see more details",
                font=ctk.CTkFont(size=11),
                text_color="gray",
                anchor="w"
            )
            tip_label.pack(fill="x", pady=(0, 10), padx=10)
            
            # Create expandable cards for each recommendation
            for i, rec in enumerate(recommendations, 1):
                card = ExpandableMovieCard(
                    self.results_scrollable_frame,
                    rec,
                    i,
                    gui_instance=self  # Pass the GUI instance
                )
                card.pack(fill="x", pady=5, padx=5)
                self.recommendation_widgets.append(card)
            
            # Store recommendations
            self.current_recommendations = recommendations
    
    def clear_recommendation_widgets(self):
        """Clear all recommendation widgets"""
        for widget in self.recommendation_widgets:
            widget.destroy()
        self.recommendation_widgets.clear()
        
        # Clear all widgets in scrollable frame
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()
    
    def analyze_sentiment(self):
        """Analyze sentiment of the review"""
        review = self.review_textbox.get("1.0", "end").strip()
        
        if not review or review == "Enter your movie review here...":
            self.show_message("Please enter a review to analyze", "error")
            return
        
        # Analyze sentiment
        sentiment, confidence = self.recommender.analyze_sentiment(review)
        
        # Update UI
        if sentiment.lower() == "positive":
            self.sentiment_result.configure(
                text="😊 POSITIVE",
                text_color=("#2e7d32", "#66bb6a")
            )
        else:
            self.sentiment_result.configure(
                text="😔 NEGATIVE",
                text_color=("#c62828", "#ef5350")
            )
        
        self.confidence_label.configure(
            text=f"Confidence: {confidence * 100:.1f}%"
        )
        self.confidence_bar.set(confidence)
    
    def get_movie_info(self):
        """Get detailed movie information"""
        movie_title = self.info_entry.get().strip()
        
        if not movie_title:
            self.show_message("Please enter a movie title", "error")
            return
        
        # Clear previous info
        self.info_textbox.configure(state="normal")
        self.info_textbox.delete("1.0", "end")
        self.info_textbox.insert("1.0", "🔍 Fetching movie information...\n")
        self.info_textbox.configure(state="disabled")
        self.root.update()
        
        # Get movie info
        movie_info = self.recommender.get_movie_info(movie_title)
        
        self.info_textbox.configure(state="normal")
        self.info_textbox.delete("1.0", "end")
        
        if not movie_info:
            self.info_textbox.insert("1.0", f"❌ Movie '{movie_title}' not found in database.\n\n"
                                            "Please check the spelling and try again.")
        else:
            output = f"🎬 {movie_info.get('title', 'N/A')}\n"
            output += "=" * 60 + "\n\n"
            
            if 'overview' in movie_info and movie_info['overview']:
                output += f"📝 Overview:\n{movie_info['overview']}\n\n"
            
            if 'genres' in movie_info and movie_info['genres']:
                output += f"🎭 Genres: {movie_info['genres']}\n\n"
            
            if 'vote_average' in movie_info:
                output += f"⭐ Rating: {movie_info['vote_average']}/10\n"
            
            if 'vote_count' in movie_info:
                output += f"🗳️ Vote Count: {movie_info['vote_count']:,}\n\n"
            
            if 'release_date' in movie_info and movie_info['release_date']:
                output += f"📅 Release Date: {movie_info['release_date']}\n\n"
            
            if 'runtime' in movie_info and movie_info['runtime']:
                output += f"⏱️ Runtime: {movie_info['runtime']} minutes\n\n"
            
            if 'cast' in movie_info and movie_info['cast']:
                output += f"🎭 Cast: {movie_info['cast']}\n\n"
            
            if 'director' in movie_info and movie_info['director']:
                output += f"🎬 Director: {movie_info['director']}\n"
        
        self.info_textbox.insert("end", output)
        self.info_textbox.configure(state="disabled")
    
    def setup_metrics_tab(self):
        """Setup the model metrics evaluation tab"""
        tab = self.tabview.tab("📊 Model Metrics")
        
        # Header
        header_frame = ctk.CTkFrame(tab, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Model Evaluation Metrics",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack()
        
        desc_label = ctk.CTkLabel(
            header_frame,
            text="View accuracy, precision, recall, F1 score, and other performance metrics",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc_label.pack(pady=(5, 15))
        
        # Button to load metrics
        button_frame = ctk.CTkFrame(tab, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.metrics_button = ctk.CTkButton(
            button_frame,
            text="Evaluate Models",
            command=self.load_metrics,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            width=200
        )
        self.metrics_button.pack()
        
        # Scrollable frame for results
        self.metrics_scroll = ctk.CTkScrollableFrame(tab, corner_radius=15)
        self.metrics_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Placeholder text
        self.metrics_text = ctk.CTkLabel(
            self.metrics_scroll,
            text="Click 'Evaluate Models' to see performance metrics",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.metrics_text.pack(pady=50)
    
    def load_metrics(self):
        """Load and display evaluation metrics"""
        self.metrics_button.configure(state="disabled", text="Evaluating...")
        self.metrics_text.configure(text="Evaluating models... This may take a moment.")
        
        def evaluate():
            try:
                evaluator = ModelEvaluator(self.recommender)
                metrics = evaluator.get_all_metrics()
                
                # Update UI in main thread
                self.root.after(0, lambda: self.display_metrics(metrics))
            except Exception as e:
                self.root.after(0, lambda: self.display_metrics_error(str(e)))
        
        threading.Thread(target=evaluate, daemon=True).start()
    
    def display_metrics(self, metrics):
        """Display metrics in the GUI"""
        self.metrics_button.configure(state="normal", text="Evaluate Models")
        
        # Clear previous content
        for widget in self.metrics_scroll.winfo_children():
            widget.destroy()
        
        # Sentiment Analysis Metrics
        if metrics.get('sentiment_analysis') and 'error' not in metrics['sentiment_analysis']:
            sa = metrics['sentiment_analysis']
            
            sa_frame = ctk.CTkFrame(self.metrics_scroll, corner_radius=15)
            sa_frame.pack(fill="x", padx=10, pady=10)
            
            sa_title = ctk.CTkLabel(
                sa_frame,
                text="📊 Sentiment Analysis Metrics",
                font=ctk.CTkFont(size=18, weight="bold")
            )
            sa_title.pack(pady=(15, 10))
            
            info_text = f"Model Type: {sa['model_type']} | Test Samples: {sa['total_samples']}"
            info_label = ctk.CTkLabel(sa_frame, text=info_text, font=ctk.CTkFont(size=12), text_color="gray")
            info_label.pack(pady=(0, 15))
            
            # Overall metrics
            overall_frame = ctk.CTkFrame(sa_frame, fg_color="transparent")
            overall_frame.pack(fill="x", padx=20, pady=10)
            
            ctk.CTkLabel(overall_frame, text="Overall Metrics", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
            
            metrics_grid = ctk.CTkFrame(overall_frame, fg_color="transparent")
            metrics_grid.pack(fill="x", pady=10)
            
            for label, value in [
                ("Accuracy", f"{sa['accuracy']:.2f}%"),
                ("Macro Precision", f"{sa['macro_precision']:.2f}%"),
                ("Macro Recall", f"{sa['macro_recall']:.2f}%"),
                ("Macro F1", f"{sa['macro_f1']:.2f}%")
            ]:
                metric_card = ctk.CTkFrame(metrics_grid, corner_radius=10)
                metric_card.pack(side="left", fill="both", expand=True, padx=5)
                
                ctk.CTkLabel(metric_card, text=label, font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(10, 5))
                ctk.CTkLabel(metric_card, text=value, font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(0, 10))
            
            # Confusion Matrix
            cm_frame = ctk.CTkFrame(sa_frame, fg_color="transparent")
            cm_frame.pack(fill="x", padx=20, pady=10)
            
            ctk.CTkLabel(cm_frame, text="Confusion Matrix", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10, 5))
            
            cm = sa['confusion_matrix']
            cm_text = f"True Positive: {cm['true_positive']} | False Positive: {cm['false_positive']}\n"
            cm_text += f"False Negative: {cm['false_negative']} | True Negative: {cm['true_negative']}"
            
            cm_label = ctk.CTkLabel(cm_frame, text=cm_text, font=ctk.CTkFont(size=12), justify="left")
            cm_label.pack(anchor="w", pady=5)
        
        # Recommendation System Metrics
        if metrics.get('recommendation_system') and 'error' not in metrics['recommendation_system']:
            rs = metrics['recommendation_system']
            
            rs_frame = ctk.CTkFrame(self.metrics_scroll, corner_radius=15)
            rs_frame.pack(fill="x", padx=10, pady=10)
            
            rs_title = ctk.CTkLabel(
                rs_frame,
                text="🎬 Recommendation System Metrics",
                font=ctk.CTkFont(size=18, weight="bold")
            )
            rs_title.pack(pady=(15, 10))
            
            info_text = f"Model Type: {rs['model_type']} | Test Movies: {rs['num_test_movies']} | K: {rs['k']}"
            info_label = ctk.CTkLabel(rs_frame, text=info_text, font=ctk.CTkFont(size=12), text_color="gray")
            info_label.pack(pady=(0, 15))
            
            # Performance metrics
            perf_frame = ctk.CTkFrame(rs_frame, fg_color="transparent")
            perf_frame.pack(fill="x", padx=20, pady=10)
            
            ctk.CTkLabel(perf_frame, text="Performance Metrics", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
            
            metrics_grid = ctk.CTkFrame(perf_frame, fg_color="transparent")
            metrics_grid.pack(fill="x", pady=10)
            
            for label, value in [
                (f"Precision@{rs['k']}", f"{rs['precision_at_k']:.2f}%"),
                (f"Recall@{rs['k']}", f"{rs['recall_at_k']:.2f}%"),
                (f"F1@{rs['k']}", f"{rs['f1_at_k']:.2f}%"),
                ("Diversity", f"{rs['diversity']:.2f}%"),
                ("Coverage", f"{rs['coverage']:.2f}%")
            ]:
                metric_card = ctk.CTkFrame(metrics_grid, corner_radius=10)
                metric_card.pack(side="left", fill="both", expand=True, padx=5)
                
                ctk.CTkLabel(metric_card, text=label, font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(10, 5))
                ctk.CTkLabel(metric_card, text=value, font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(0, 10))
    
    def display_metrics_error(self, error_msg):
        """Display error message"""
        self.metrics_button.configure(state="normal", text="Evaluate Models")
        
        for widget in self.metrics_scroll.winfo_children():
            widget.destroy()
        
        error_label = ctk.CTkLabel(
            self.metrics_scroll,
            text=f"❌ Error: {error_msg}",
            font=ctk.CTkFont(size=13),
            text_color="red"
        )
        error_label.pack(pady=50)
    
    def on_search_key_release(self, entry_widget, context):
        """Handle key release in search entry - show autocomplete suggestions"""
        query = entry_widget.get().strip()
        
        # Close autocomplete if query is too short
        if len(query) < 2:
            self.close_autocomplete()
            return
        
        # Get suggestions
        suggestions = self.recommender.get_movie_suggestions(query, limit=8)
        
        if suggestions:
            if context == "search":
                self.search_suggestions = suggestions
            else:
                self.info_suggestions = suggestions
            self.show_autocomplete(entry_widget, suggestions, context)
        else:
            self.close_autocomplete()
    
    def show_autocomplete(self, entry_widget, suggestions: List[str], context: str):
        """Display autocomplete dropdown below the entry widget"""
        # Close existing autocomplete window
        self.close_autocomplete()
        
        # Create new toplevel window
        self.autocomplete_window = ctk.CTkToplevel(self.root)
        self.autocomplete_window.withdraw()  # Hide initially
        
        # Remove window decorations
        self.autocomplete_window.overrideredirect(True)
        self.autocomplete_window.attributes("-alpha", 0.96)
        
        # Create frame for suggestions
        suggestions_frame = ctk.CTkScrollableFrame(
            self.autocomplete_window,
            fg_color=("#d0d0d0", "#2b2b2b"),
            corner_radius=10,
            height=min(len(suggestions) * 40, 320)
        )
        suggestions_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Add suggestion buttons
        for suggestion in suggestions:
            btn = ctk.CTkButton(
                suggestions_frame,
                text=suggestion,
                font=ctk.CTkFont(size=13),
                height=35,
                corner_radius=8,
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("#3b8ed0", "#1f6aa5"),
                anchor="w",
                command=lambda s=suggestion: self.select_suggestion(entry_widget, s, context)
            )
            btn.pack(fill="x", padx=3, pady=2)
        
        # Position the window below the entry widget
        self.position_autocomplete(entry_widget)
        
        # Show the window
        self.autocomplete_window.deiconify()
        
        # Bind click outside to close
        self.autocomplete_window.bind("<FocusOut>", lambda e: self.close_autocomplete())
        
    def position_autocomplete(self, entry_widget):
        """Position autocomplete window below the entry widget"""
        # Update to get accurate coordinates
        entry_widget.update_idletasks()
        
        # Get entry widget position
        x = entry_widget.winfo_rootx()
        y = entry_widget.winfo_rooty() + entry_widget.winfo_height()
        width = entry_widget.winfo_width()
        
        # Set window position and size
        self.autocomplete_window.geometry(f"{width}x320+{x}+{y}")
    
    def select_suggestion(self, entry_widget, suggestion: str, context: str):
        """Handle suggestion selection"""
        # Clear and set entry value
        entry_widget.delete(0, "end")
        entry_widget.insert(0, suggestion)
        
        # Close autocomplete
        self.close_autocomplete()
        
        # Auto-trigger search/info based on context
        if context == "search":
            self.search_movies()
        else:
            self.get_movie_info()
    
    def close_autocomplete(self):
        """Close the autocomplete window if it exists"""
        if self.autocomplete_window:
            try:
                self.autocomplete_window.destroy()
            except:
                pass
            self.autocomplete_window = None
    
    def show_message(self, message, msg_type="info"):
        """Show a message dialog"""
        # This could be enhanced with a custom dialog
        print(f"{msg_type.upper()}: {message}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    print("=" * 70)
    print("Starting Movie Recommender GUI Application...")
    print("=" * 70)
    print()
    
    try:
        app = MovieRecommenderGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
