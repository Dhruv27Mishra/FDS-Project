# Algorithms Used in Movie Recommendation System

This document details all the algorithms and techniques used for each function in the Movie Recommendation System with Sentiment Analysis.

---

## 1. Movie Recommendations (`recommend_movies`)

### Primary Algorithm: **Content-Based Filtering with Cosine Similarity**

#### Algorithms & Techniques:

1. **Count Vectorization (Bag of Words)**
   - **Algorithm**: `CountVectorizer` from scikit-learn
   - **Purpose**: Converts movie features (director, actors, genres) into numerical vectors
   - **Process**: 
     - Combines director name, actor names, and genres into a single text string (`comb` column)
     - Creates a document-term matrix where each movie is represented as a vector
     - Each dimension represents a unique word/feature from the combined text
   - **Location**: `movie_recommender.py` line 40-41

2. **Cosine Similarity**
   - **Algorithm**: `cosine_similarity` from scikit-learn
   - **Purpose**: Measures similarity between movies based on their feature vectors
   - **Formula**: 
     ```
     cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
     ```
   - **Process**:
     - Computes pairwise cosine similarity between all movies
     - Creates a similarity matrix (N×N) where N is the number of movies
     - Values range from 0 (no similarity) to 1 (identical)
   - **Location**: `movie_recommender.py` line 42

3. **Sorting & Ranking**
   - **Algorithm**: Python's built-in `sorted()` with custom key function
   - **Purpose**: Ranks movies by similarity score in descending order
   - **Process**: 
     - Extracts similarity scores for the input movie
     - Sorts all movies by their similarity score
     - Returns top N recommendations (excluding the movie itself)
   - **Location**: `movie_recommender.py` line 100-106

**Time Complexity**: O(N²) for similarity matrix computation, O(N log N) for sorting

---

## 2. Sentiment Analysis (`analyze_sentiment`)

### Primary Algorithm: **Multinomial Naive Bayes Classifier**

#### Algorithms & Techniques:

1. **TF-IDF Vectorization**
   - **Algorithm**: `TfidfVectorizer` from scikit-learn
   - **Purpose**: Converts text reviews into numerical feature vectors
   - **Components**:
     - **Term Frequency (TF)**: How often a word appears in a document
     - **Inverse Document Frequency (IDF)**: How rare/common a word is across all documents
   - **Formula**:
     ```
     TF-IDF(t, d) = TF(t, d) × IDF(t)
     IDF(t) = log(N / df(t))
     ```
     where:
     - N = total number of documents
     - df(t) = number of documents containing term t
   - **Features**:
     - Uses unigrams and bigrams (ngram_range=(1, 2))
     - Removes stop words (common words like "the", "and", etc.)
     - Converts to lowercase
     - Strips accents
   - **Location**: Pre-trained model loaded from `tranform.pkl`

2. **Multinomial Naive Bayes**
   - **Algorithm**: `MultinomialNB` from scikit-learn
   - **Purpose**: Classifies reviews as positive (1) or negative (0)
   - **Mathematical Foundation**:
     - Based on Bayes' Theorem with "naive" independence assumption
     - Formula:
     ```
     P(class|features) = P(class) × ∏ P(feature_i|class) / P(features)
     ```
   - **Why Multinomial?**: 
     - Suitable for discrete count data (word frequencies)
     - Works well with TF-IDF vectors
   - **Training**: Pre-trained on movie reviews dataset (6918 reviews)
   - **Location**: Pre-trained model loaded from `nlp_model.pkl`

3. **Probability Estimation**
   - **Algorithm**: `predict_proba()` method
   - **Purpose**: Provides confidence scores for predictions
   - **Process**:
     - Returns probability distribution over classes [P(negative), P(positive)]
     - Confidence = max(probabilities)
   - **Location**: `movie_recommender.py` line 144-147

**Accuracy**: ~98.77% (as shown in training results)

---

## 3. Movie Search (`get_movie_suggestions`)

### Primary Algorithm: **String Matching with Substring Search**

#### Algorithms & Techniques:

1. **Substring Matching**
   - **Algorithm**: Pandas string operations with `.str.contains()`
   - **Purpose**: Finds movies whose titles contain the search query
   - **Process**:
     - Converts both query and movie titles to lowercase for case-insensitive search
     - Uses regex pattern matching to find partial matches
     - Returns all movies where title contains the query substring
   - **Location**: `movie_recommender.py` line 73-75

2. **Result Limiting**
   - **Algorithm**: Python list slicing
   - **Purpose**: Limits results to top N matches
   - **Process**: Returns first `limit` number of matches
   - **Location**: `movie_recommender.py` line 77

**Time Complexity**: O(N × M) where N = number of movies, M = average title length

---

## 4. Movie Details Retrieval (`get_movie_info`)

### Primary Algorithm: **Database Lookup with Exact Match**

#### Algorithms & Techniques:

1. **Exact String Matching**
   - **Algorithm**: Pandas boolean indexing with `.str.lower()` comparison
   - **Purpose**: Finds exact movie match in database
   - **Process**:
     - Converts both query and database titles to lowercase
     - Performs exact match (not substring)
     - Returns first matching row
   - **Location**: `movie_recommender.py` line 158-163

2. **Data Extraction**
   - **Algorithm**: Pandas DataFrame `.iloc[]` indexing
   - **Purpose**: Extracts specific columns from matched row
   - **Location**: `movie_recommender.py` line 163-174

**Time Complexity**: O(N) where N = number of movies

---

## Summary Table

| Function | Primary Algorithm | Supporting Algorithms | Complexity |
|----------|------------------|----------------------|------------|
| **Movie Recommendations** | Cosine Similarity | Count Vectorization, Sorting | O(N²) |
| **Sentiment Analysis** | Multinomial Naive Bayes | TF-IDF Vectorization, Probability Estimation | O(V) where V = vocabulary size |
| **Movie Search** | Substring Matching | String Operations, List Slicing | O(N × M) |
| **Movie Details** | Exact String Match | Boolean Indexing, Data Extraction | O(N) |

---

## Data Structures Used

1. **Pandas DataFrame**: Stores movie data (6116 movies)
2. **NumPy Arrays**: Stores similarity matrices and feature vectors
3. **Scipy Sparse Matrices**: Efficient storage of count matrices (sparse format)
4. **Python Lists/Dictionaries**: For storing recommendations and results

---

## Model Training Details

### Sentiment Analysis Model:
- **Training Dataset**: 6918 movie reviews (from `datasets/reviews.txt`)
- **Train-Test Split**: 80-20 split
- **Feature Extraction**: TF-IDF with unigrams and bigrams
- **Model**: Multinomial Naive Bayes
- **Accuracy**: 98.77% on test set
- **Precision**: 0.85
- **Recall**: 0.80
- **F1 Score**: 0.83

### Recommendation System:
- **Dataset**: 6116 movies with features (director, actors, genres)
- **Feature Combination**: All features combined into single text string
- **Similarity Matrix**: Pre-computed for all movie pairs
- **Storage**: Similarity matrix computed at runtime (can be cached for performance)

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Naive Bayes: https://en.wikipedia.org/wiki/Naive_Bayes_classifier

