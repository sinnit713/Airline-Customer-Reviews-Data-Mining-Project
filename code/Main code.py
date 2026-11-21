"""
Airline Customer Review Analysis Pipeline

This script performs end to end data analysis including:
- Data cleaning and preprocessing
- Text feature engineering (TF-IDF, sentiment)
- Descriptive analytics and visualisation
- Predictive modelling (Numerical, Text-based, Combined)

The goal is to predict whether a customer recommends an airline
based on structured and unstructured review data.

Reproducibility:
- Fixed random_state used in model training
- Deterministic preprocessing pipeline

Dependencies:
- pandas, numpy, sklearn, nltk, matplotlib, textblob, scipy
"""

import re
import sys
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from textblob import TextBlob

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure proper encoding to avoid Unicode errors during printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Download required NLTK resources
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Required columns for analysis
# These are validated to ensure dataset consistency
REQUIRED_COLUMNS = [
    "Review",
    "Overall_Rating",
    "Recommended",
    "Seat Comfort",
    "Cabin Staff Service",
    "Food & Beverages",
    "Ground Service",
    "Inflight Entertainment",
    "Wifi & Connectivity",
    "Value For Money",
]

# Numerical features used for modelling
# Includes overall rating and service-specific attributes
NUMERIC_FEATURES = [
    "Overall_Rating",
    "Seat Comfort",
    "Cabin Staff Service",
    "Food & Beverages",
    "Ground Service",
    "Inflight Entertainment",
    "Wifi & Connectivity",
    "Value For Money",
]


def load_data(path):
    """
    Load dataset from an Excel file.

    Parameters:
    path (str): File path to the dataset.

    Returns:
    pd.DataFrame: Loaded dataset.

    Notes:
    - Uses openpyxl engine for compatibility with Excel files.
    - Assumes dataset structure matches expected schema.
    """
    # Load dataset into DataFrame
    df = pd.read_excel(path, engine="openpyxl")
    return df


def validate_columns(df):
    """
    Validate presence of required columns in the dataset.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Raises:
    ValueError: If any required column is missing.

    Purpose:
    - Ensures dataset integrity before processing
    - Prevents downstream errors during modelling
    """
    # Identify missing required columns
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    # Raise error if any required column is absent
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def initial_inspection(df):
    """
    Perform initial exploratory inspection of the dataset.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Outputs:
    - Displays head of dataset
    - Prints data types and non-null counts
    - Shows distribution of key variable (Overall Rating)

    Purpose:
    - Understand data structure and quality
    - Identify inconsistencies before cleaning
    """
    print("\n DATA HEAD: ")
    print(df.head())

    print("\nDATA INFO:")
    # Provides column types and missing values overview
    df.info()

    print("\nOVERALL RATING UNIQUE: ")
    # Helps detect invalid or inconsistent entries
    print(df["Overall_Rating"].unique())

    print("\nOVERALL RATING DISTRIBUTION: ")
    # Shows frequency distribution including missing values
    print(df["Overall_Rating"].value_counts(dropna=False))


def clean_overall_rating(df):
    """
    Clean and convert the 'Overall_Rating' column to numeric format.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Returns:
    pd.DataFrame: Updated dataset with cleaned rating column.

    Purpose:
    - Ensures numerical consistency for modelling
    - Handles invalid or non-numeric entries safely

    Assumption:
    - Non-convertible values are treated as missing (NaN)
    """
    # Convert rating to numeric; invalid values become NaN
    df["Overall_Rating"] = pd.to_numeric(df["Overall_Rating"], errors="coerce")

    print("\nOVERALL RATING AFTER CLEANING: ")
    print("Dtype:", df["Overall_Rating"].dtype)
    print("Missing values:", df["Overall_Rating"].isnull().sum())
    print(df["Overall_Rating"].value_counts(dropna=False))

    return df


def analyze_target(df):
    """
    Analyse the distribution and structure of the target variable 'Recommended'.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Purpose:
    - Understand unique values and class distribution
    - Detect inconsistencies (e.g., Yes/yes/YES)
    - Identify class imbalance before modelling
    """

    print("\nTARGET VARIABLE: ")
    # Display unique values to check for inconsistencies
    print(df["Recommended"].unique())
    #Display full distribution including missing values
    print(df["Recommended"].value_counts(dropna=False))


def contradiction_analysis(df):
    """
    Identify contradictions between ratings and recommendation behaviour.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Purpose:
    - Detects behavioural inconsistencies
    - Highlights non-linear decision patterns

    Examples:
    - High rating but not recommended
    - Low rating but recommended
    """
    target = df["Recommended"].astype(str).str.strip().str.lower()
    # Cases where high rating does not lead to recommendation
    high_no = df[(df["Overall_Rating"] >= 7) & (target == "no")].shape[0]
    # Cases where low rating still results in recommendation
    low_yes = df[(df["Overall_Rating"] <= 4) & (target == "yes")].shape[0]

    print("\nCONTRADICTIONS: ")
    print("High rating but NO:", high_no)
    print("Low rating but YES:", low_yes)


def check_duplicates(df):
    """
    Check for duplicate records in the dataset.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Purpose:
    - Identifies redundant records
    - Helps ensure data quality and reliability
    """
    print("\nDUPLICATES: ")
    print("Exact duplicates: ", df.duplicated().sum())


def text_analysis(df):
    """
    Perform basic analysis of review text length.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Returns:
    pd.DataFrame: Dataset with review length feature added.

    Purpose:
    - Captures customer engagement level
    - Longer reviews often indicate stronger sentiment or dissatisfaction
    """
    # Compute character length of each review
    print("\nREVIEW LENGTH STATS: ")
    print(df["Review"].astype(str).str.len().describe())


def clean_text(text):
    """
    Clean and normalize review text for NLP processing.

    Parameters:
    text (str): Raw review text.

    Returns:
    str: Cleaned and processed text.

    Steps:
    - Convert to lowercase
    - Remove non-alphanumeric characters
    - Remove stopwords (excluding negations)
    - Remove domain-specific noise

    Purpose:
    - Standardises text for TF-IDF vectorisation
    - Preserves sentiment-relevant terms (e.g., 'not', 'never')
    """
    # Convert to string and lowercase for consistency
    text = str(text).lower()
    # Remove non-alphanumeric characters (retain spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.encode("ascii", "ignore").decode()
    return text


def apply_text_cleaning(df):
    """
    Apply basic text cleaning to review data.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Returns:
    pd.DataFrame: Dataset with cleaned review text.

    Purpose:
    - Standardises text for downstream NLP tasks
    - Removes noise and ensures consistency

    Notes:
    - Uses the 'clean_text' function for preprocessing
    - Removes common dataset-specific phrase "trip verified"
    """
    # Apply base text cleaning function
    df["clean_review"] = df["Review"].astype(str).apply(clean_text)
    # Remove frequently occurring but non-informative phrase
    # This prevents bias in TF-IDF features
    df["clean_review"] = df["clean_review"].str.replace("trip verified", "", regex=False)
    return df


def remove_stopwords(df):
    """
    Remove stopwords from cleaned review text while preserving negations.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Returns:
    pd.DataFrame: Dataset with stopwords removed.

    Purpose:
    - Reduces noise in text data
    - Retains important sentiment-carrying words

    Key Design Choice:
    - Negation words (e.g., 'not', 'never') are preserved as they significantly impact sentiment polarity
    """
    # Load standard English stopwords
    stop_words = set(stopwords.words("english"))
    # Preserve negation terms as they invert sentiment meaning
    negation_words = {"not", "no", "nor", "never"}
    filtered_stopwords = stop_words - negation_words

    def filter_text(text):
        """
        Remove stopwords from a single text string.
        """
        words = str(text).split()
        # Remove stopwords while keeping meaningful tokens
        words = [word for word in words if word not in filtered_stopwords]
        return " ".join(words)
    # Apply stopword removal to all reviews
    df["clean_review"] = df["clean_review"].apply(filter_text)
    return df


def add_features(df):
    """
    Create additional text-based features.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Returns:
    pd.DataFrame: Dataset with new features.

    Features Added:
    - review_length: Length of cleaned review text

    Purpose:
    - Captures customer engagement level
    - Longer reviews often indicate stronger sentiment or dissatisfaction
    """
    # Compute length of cleaned review text
    df["review_length"] = df["clean_review"].astype(str).apply(len)
    return df


def compare_text(df, n=5):
    """
    Compare original and cleaned review text samples.

    Parameters:
    df (pd.DataFrame): Input dataset.
    n (int): Number of samples to display.

    Purpose:
    - Provides qualitative validation of preprocessing
    - Ensures cleaning steps do not distort meaning
    """
    print("\nTEXT COMPARISON: ")
    # Limit sample size to dataset size
    sample_size = min(n, len(df))
    for i in range(sample_size):
        # Convert original text safely to avoid encoding issues
        original = str(df["Review"].iloc[i]).encode("ascii", "ignore").decode()
        cleaned = df["clean_review"].iloc[i]

        print("\nORIGINAL:\n", original)
        print("\nCLEANED:\n", cleaned)
        print("\n" + "-" * 80)


def convert_numeric_columns(df, columns):
    
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def prepare_target(df):
    """
    Encode target variable 'Recommended' into binary format.

    Parameters:
    df (pd.DataFrame): Input dataset

    Returns:
    pd.DataFrame: Updated dataset

    Purpose:
    - Converts Yes/No labels into numeric format (1/0)
    - Ensures compatibility with machine learning models

    Note:
    - Applied after initial analysis to preserve raw data inspection
    """

    # Normalize text and map to binary values
    df["Recommended"] = (
        df["Recommended"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    df = df.dropna(subset=["Recommended"]).copy()
    df["Recommended"] = df["Recommended"].astype(int)
    return df


def build_text_features(df):
    """
    Generate TF-IDF features from cleaned review text.

    Parameters:
    df (pd.DataFrame): Input dataset with cleaned review text.

    Returns:
    tfidf (TfidfVectorizer): Fitted TF-IDF vectorizer
    X_tfidf (sparse matrix): Transformed TF-IDF feature matrix

    Purpose:
    - Converts textual data into numerical feature representation
    - Captures word importance based on frequency and uniqueness

    Design Choices:
    - max_features=3000 → Controls dimensionality to avoid overfitting
    - ngram_range=(1,2) → Captures both single words and short phrases
    - min_df=5 → Removes rare words (noise reduction)
    """
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
    )
    # Fit TF-IDF model and transform text into feature matrix
    X_tfidf = tfidf.fit_transform(df["clean_review"])

    print("\nTF-IDF SHAPE: ")
    print(X_tfidf.shape)

    feature_names = tfidf.get_feature_names_out()
    print("\nSAMPLE FEATURES: ")
    print(feature_names[:20])

    return tfidf, X_tfidf


def add_sentiment_features(df):
    """
    Add sentiment-based features using VADER and TextBlob.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Returns:
    pd.DataFrame: Dataset with sentiment features added.

    Features Added:
    - sentiment_score (VADER compound score)
    - textblob_polarity (TextBlob polarity score)

    Purpose:
    - Capture emotional tone of customer reviews
    - Complement TF-IDF features with sentiment signals

    Methodological Note:
    - VADER is rule-based and sensitive to intensity and negation
    - TextBlob provides a lexicon-based polarity measure
    - Using both allows comparative sentiment analysis
    """
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    df["sentiment_score"] = df["Review"].astype(str).apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    print("\n SENTIMENT SAMPLE: ")
    print(df[["clean_review", "sentiment_score"]].head())
    # Compute TextBlob polarity score
    df["textblob_polarity"] = df["Review"].astype(str).apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    print("\n TEXTBLOB SENTIMENT SAMPLE: ")
    print(df[["clean_review", "sentiment_score", "textblob_polarity"]].head())
    
    print("\n SENTIMENT COMPARISON (VADER vs TEXTBLOB): ")
    print(df[["sentiment_score", "textblob_polarity"]].corr())

    return df


def show_descriptive_statistics(df):
    """
    Display descriptive statistics for all features.

    Parameters:
    df (pd.DataFrame): Input dataset

    Purpose:
    - Provides summary statistics for numerical and categorical variables
    - Ensures safe printing (avoids encoding issues)
    """
    print("\n NUMERICAL DESCRIPTIVE STATISTICS: ")
    num_df = df.select_dtypes(include=[np.number])

    print(num_df.describe())

    print("\n MISSING VALUES (NUMERICAL): ")
    print(num_df.isnull().sum())


def create_visualizations(df):
    """
    Generate key visualisations for analysis.

    Parameters:
    df (pd.DataFrame): Input dataset

    Purpose:
    - Supports descriptive analytics (Sections 4.a and 4.b)
    - Helps identify patterns, correlations, and contradictions
    """
    plt.figure()
    df["Overall_Rating"].hist()
    plt.title("Distribution of Overall Rating")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    rating_recommend = pd.crosstab(
        df["Overall_Rating"], df["Recommended"], normalize="index"
    )
    rating_recommend.plot(kind="bar", stacked=True)
    plt.title("Overall Rating vs Recommendation")
    plt.xlabel("Overall Rating")
    plt.ylabel("Proportion")
    plt.legend(title="Recommended")
    plt.tight_layout()
    plt.show()

    vfm_recommend = pd.crosstab(
        df["Value For Money"], df["Recommended"], normalize="index"
    )
    vfm_recommend.plot(kind="bar", stacked=True)
    plt.title("Value for Money vs Recommendation")
    plt.xlabel("Value For Money Rating")
    plt.ylabel("Proportion")
    plt.legend(title="Recommended")
    plt.tight_layout()
    plt.show()

    corr = df[NUMERIC_FEATURES].corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.show()

    service_cols = NUMERIC_FEATURES[1:]
    yes_avg = df[df["Recommended"] == 1][service_cols].mean()
    no_avg = df[df["Recommended"] == 0][service_cols].mean()

    comparison = pd.DataFrame(
        {"Recommended Yes": yes_avg, "Recommended No": no_avg}
    ).sort_values(by="Recommended Yes")

    comparison.plot(kind="barh", figsize=(8, 6))
    plt.title("Service Ratings: Recommended vs Not Recommended")
    plt.xlabel("Average Rating")
    plt.ylabel("Service Category")
    plt.tight_layout()
    plt.show()

    plt.figure()
    df.boxplot(column="sentiment_score", by="Recommended")
    plt.title("Sentiment Distribution by Recommendation")
    plt.suptitle("")
    plt.xlabel("Recommended")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.scatter(df["sentiment_score"], df["textblob_polarity"], alpha=0.3)
    plt.title("VADER vs TextBlob Sentiment")
    plt.xlabel("VADER Score")
    plt.ylabel("TextBlob Polarity")
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.scatter(
        df["sentiment_score"],
        df["Overall_Rating"],
        c=df["Recommended"],
        cmap="coolwarm",
        alpha=0.3
    )
    plt.colorbar(label="Recommended (0 = No, 1 = Yes)")
    plt.title("Contradictions: Sentiment vs Rating vs Recommendation")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Overall Rating")
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    df.boxplot(column="review_length", by="Recommended")
    plt.title("Review Length vs Recommendation")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    df[df["Recommended"] == 1]["sentiment_score"].hist(alpha=0.5, label="Recommended YES")
    df[df["Recommended"] == 0]["sentiment_score"].hist(alpha=0.5, label="Recommended NO")
    plt.legend()
    plt.title("Sentiment Distribution by Recommendation")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def train_numerical_models(df):
    """
    Train and evaluate supervised models using numerical features.

    Parameters:
    df (pd.DataFrame): Input dataset with cleaned and processed features.

    Returns:
    None

    Models Used:
    - Logistic Regression → interpretable baseline model
    - Random Forest → captures non-linear relationships

    Purpose:
    - Predict customer recommendation using structured service attributes
    - Compare performance of linear vs non-linear models

    Evaluation Metrics:
    - Accuracy
    - Confusion Matrix
    - Classification Report

    Reproducibility:
    - Fixed random_state ensures consistent results across runs
    """
    
    X = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    y = df["Recommended"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print("\n LOGISTIC REGRESSION RESULTS: ")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)

    print("\n BEST PARAMETERS: ")
    print(grid.best_params_)

    best_lr = grid.best_estimator_
    y_pred_tuned = best_lr.predict(X_test)

    print("\n TUNED LOGISTIC REGRESSION: ")
    print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n RANDOM FOREST RESULTS: ")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    feature_importance = pd.Series(rf.feature_importances_, index=NUMERIC_FEATURES)
    feature_importance = feature_importance.sort_values(ascending=False)
    print("\n FEATURE IMPORTANCE: ")
    print(feature_importance)

    return best_lr


def time_based_test(df, model):
    """
    Perform time-based validation to test model robustness.

    Parameters:
    df (pd.DataFrame): Input dataset.

    Returns:
    None

    Purpose:
    - Evaluate model performance on future data
    - Simulates real-world deployment scenario

    Method:
    - Train model on earlier data
    - Test model on more recent data

    Insight:
    - Helps assess temporal stability and generalisability
    """
    if "Review Date" not in df.columns:
        print("\n TIME-BASED TEST SKIPPED: ")
        print("Column 'Review Date' not found.")
        return

    df = df.copy()
    df["Review Date"] = pd.to_datetime(df["Review Date"], errors="coerce")
    df_time = df.dropna(subset=["Review Date"]).sort_values(by="Review Date")

    if len(df_time) < 10:
        print("\n TIME-BASED TEST SKIPPED: ")
        print("Not enough valid dated rows.")
        return
    # Split data: 70% past (train), 30% recent (test)
    split_index = int(0.7 * len(df_time))
    train_df = df_time.iloc[:split_index]
    test_df = df_time.iloc[split_index:]

    X_train = train_df[NUMERIC_FEATURES].fillna(train_df[NUMERIC_FEATURES].median())
    y_train = train_df["Recommended"]
    X_test = test_df[NUMERIC_FEATURES].fillna(train_df[NUMERIC_FEATURES].median())
    y_test = test_df["Recommended"]
    
    # Train Logistic Regression model
    model.fit(X_train, y_train)
    y_pred_time = model.predict(X_test)

    print("\n TIME-BASED TEST (LR): ")
    print("Accuracy:", accuracy_score(y_test, y_pred_time))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_time))


def train_text_model(df, X_tfidf, tfidf):
    y = df["Recommended"].values

    X_text = hstack(
        [
            X_tfidf,
            np.array(df["sentiment_score"]).reshape(-1, 1),
            np.array(df["textblob_polarity"]).reshape(-1, 1),
            np.array(df["review_length"]).reshape(-1, 1),
        ]
    )

    indices = np.arange(len(df))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_text, y, indices,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    text_lr = LogisticRegression(max_iter=1000)
    text_lr.fit(X_train, y_train)
    y_pred_text = text_lr.predict(X_test)
    
    # ================================
    # MISCLASSIFIED REVIEWS
    # ================================
    mis_idx = np.where(y_test != y_pred_text)[0]

    print("\n MISCLASSIFIED REVIEWS : 10 reviews full analysis: ")

    for i in mis_idx[:10]:
        real_index = idx_test[i]

        print("\nREVIEW:")
        print(df.iloc[real_index]["Review"])

        print("\nSentiment Score:", df.iloc[real_index]["sentiment_score"])
        print("TextBlob Polarity:", df.iloc[real_index]["textblob_polarity"])

        print("\nActual:", y_test[i], "Predicted:", y_pred_text[i])
        print("-"*80)
    

    print("\n TEXT-BASED LOGISTIC REGRESSION: ")
    print("Accuracy:", accuracy_score(y_test, y_pred_text))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_text))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_text))

    # ================================
    # INTERPRETABILITY (TOP WORDS)
    # ================================

    feature_names = tfidf.get_feature_names_out()
    coefficients = text_lr.coef_[0][:len(feature_names)]

    top_positive = np.argsort(coefficients)[-10:]
    top_negative = np.argsort(coefficients)[:10]

    print("\n TOP POSITIVE WORDS: ")
    print(feature_names[top_positive])

    print("\n TOP NEGATIVE WORDS: ")
    print(feature_names[top_negative])
    
def train_combined_model(df, X_tfidf):
    """
    Train a model using combined numerical and textual features.

    Parameters:
    df (pd.DataFrame): Input dataset
    X_tfidf (sparse matrix): TF-IDF feature matrix

    Returns:
    None

    Purpose:
    - Integrates structured (numerical) and unstructured (text) data
    - Evaluates whether combining features improves predictive performance

    Feature Groups:
    - Numerical features (service ratings)
    - Textual features (TF-IDF)
    - Sentiment features (VADER, TextBlob)
    - Behavioural feature (review length)
    """
    y = df["Recommended"].values

    # Numerical features
    X_num = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median()).values

    # Combine EVERYTHING
    X_combined = hstack([
        X_tfidf,
        np.array(df["sentiment_score"]).reshape(-1, 1),
        np.array(df["textblob_polarity"]).reshape(-1, 1),
        np.array(df["review_length"]).reshape(-1, 1),
        X_num
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n COMBINED MODEL RESULTS: ")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    """
    Execute full data analysis pipeline.

    Steps:
    1. Load data
    2. Validate structure
    3. Perform cleaning and preprocessing
    4. Generate features
    5. Train models
    6. Evaluate and interpret results
    """
    df = load_data("MGT0000_ITAG7105_new_dataset_final.xlsx")
    validate_columns(df)

    initial_inspection(df)
    df = clean_overall_rating(df)
    df = convert_numeric_columns(df, NUMERIC_FEATURES[1:])

    analyze_target(df)
    contradiction_analysis(df)
    check_duplicates(df)
    text_analysis(df)

    df = apply_text_cleaning(df)
    df = remove_stopwords(df)
    df = add_features(df)
    compare_text(df)

    tfidf, X_tfidf = build_text_features(df)
    df = add_sentiment_features(df)
    
    print("\n SENTIMENT vs RATING CORRELATION: ")
    print(df[["sentiment_score", "Overall_Rating"]].corr())
    
    show_descriptive_statistics(df)

    df = prepare_target(df)
    
    # TEXT CONTRADICTION ANALYSIS
    pos_no = df[(df["sentiment_score"] > 0.5) & (df["Recommended"] == 0)]
    neg_yes = df[(df["sentiment_score"] < -0.5) & (df["Recommended"] == 1)]

    print("\n TEXT CONTRADICTIONS: ")
    print("Positive sentiment but NO:", len(pos_no))
    print("Negative sentiment but YES:", len(neg_yes))
    
    create_visualizations(df)

    best_lr = train_numerical_models(df)
    time_based_test(df, best_lr)
    train_text_model(df, X_tfidf, tfidf)
    train_combined_model(df, X_tfidf)

if __name__ == "__main__":
    main()
