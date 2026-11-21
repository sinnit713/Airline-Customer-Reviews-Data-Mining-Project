# Airline-Customer-Reviews-Data-Mining-Project
An end to end data mining project analysing airline customer reviews using Python, covering descriptive analytics, predictive modelling, NLP based sentiment analysis, and multimodal feature integration.

# Descriptive, Predictive, and Text Analytics for Airline Customer Reviews

## Project Overview

This project investigates airline customer recommendation behaviour using descriptive analytics, predictive modelling, and text analytics. It explores how structured service ratings and unstructured passenger reviews can be combined to identify factors associated with customer recommendations.

The analysis was completed as part of **ITAG7105 – Data Mining** for the MSc Business Analytics programme at Queen’s University Belfast, GIFT City Campus.

## Research Objective

The primary objective is to predict whether a passenger recommends an airline and identify the factors influencing that decision.

The project examines three analytical dimensions:

- **Descriptive analytics:** Explore customer ratings, service quality indicators, and recommendation patterns.
- **Predictive analytics:** Develop classification models using numerical and textual features.
- **Text analytics:** Extract sentiment and contextual information from customer reviews and investigate how textual feedback relates to recommendation behaviour.

## Analytical Workflow

The project follows the CRISP-DM framework and includes the following stages:

### 1. Data Preparation
- Inspect and clean numerical variables.
- Handle missing values using median imputation.
- Preprocess textual reviews while retaining important negation terms.
- Engineer review length and sentiment features.

### 2. Descriptive Analytics
- Examine distributions and summary statistics.
- Investigate relationships between service ratings and customer recommendations.
- Analyse differences between recommended and non-recommended reviews.
- Develop visualisations to communicate analytical findings.

### 3. Predictive Modelling
The following classification approaches were evaluated:

- Logistic Regression
- Random Forest

Models were developed using numerical service-rating features and evaluated through classification metrics and confusion matrices. A time-based validation exercise was also conducted to examine performance on more recent reviews.

### 4. Text Analytics
Textual analysis incorporated:

- TF-IDF feature extraction using unigrams and bigrams.
- VADER sentiment analysis.
- TextBlob sentiment polarity.
- Review-length analysis.
- Examination of sentiment and recommendation contradictions.
- Review-level misclassification analysis.

### 5. Combined Feature Modelling
Numerical features, TF-IDF representations, sentiment scores, and review-length features were integrated into a combined Logistic Regression model to assess whether structured and unstructured information provide complementary predictive value.

## Key Findings

The reported coursework results showed the following model performance:

| Model | Feature Type | Reported Accuracy |
|---|---|---:|
| Logistic Regression | Numerical | ~95.6% |
| Random Forest | Numerical | ~95.9% |
| Logistic Regression | Text-based | ~94.9% |
| Logistic Regression | Combined | **96.49%** |

The combined model achieved the highest reported accuracy of 96.49%.

Other findings included:

- Value for Money, Cabin Staff Service, and Ground Service emerged as important numerical predictors.
- Textual feedback provided additional context, particularly around refunds, customer service failures, and service disruptions.
- Customer ratings and recommendation behaviour were not always aligned.
- Time-based validation produced approximately 94.2% accuracy, indicating some reduction in predictive performance over time.

*Note: These figures are results reported in the coursework submission. They have not been independently reproduced or revalidated for this repository.*

## Repository Structure

```text
airline-customer-reviews-data-mining/
│
├── code/
│   ├── Main code.py
│   └── README.md
│
├── data/
│   ├── Dataset.xlsx
│   └── README.md
│
├── visualisations/
│   ├── part_a/
│   │   └── README.md
│   │
│   └── part_b/
│       └── README.md
│
├── .gitignore
└── README.md
```

## Technology Stack

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Text Analytics:** NLTK, VADER, TextBlob
- **Feature Extraction:** TF-IDF
- **Visualisation:** Matplotlib
- **Data Format:** Microsoft Excel

## Repository Contents

- `code/` – Original Python source code used for the analysis.
- `data/` – Dataset supplied for the coursework.
- `visualisations/part_a/` – Visualisations for descriptive and predictive analytics.
- `visualisations/part_b/` – Visualisations for text analytics.

## Limitations

The analysis has several limitations:

- Sentiment methods may struggle with sarcasm, negation, and context-dependent expressions.
- TF-IDF captures word patterns but does not fully represent semantic meaning.
- Class imbalance and missing values may influence model behaviour.
- Customer expectations and airline service standards can change over time.
- Reported model performance should be interpreted in the context of the dataset and validation approach.

## Dataset Attribution and Usage

The dataset was supplied by the ITAG7105 module coordinator and is based on a dataset obtained from Kaggle for educational purposes.

The module coursework brief specifies that students must use the particular dataset version supplied by the coordinator. The original dataset ownership and redistribution rights do not belong to the module teaching team.

The dataset is included here for educational and portfolio documentation purposes. Its presence in this repository should not be interpreted as permission for unrestricted redistribution or commercial use. Users should verify the original dataset licence and applicable terms before reuse.

## Academic Context

**Module:** ITAG7105 – Data Mining  
**Programme:** MSc Business Analytics  
**Institution:** Queen’s University Belfast – GIFT City Campus  
**Project:** Descriptive, Predictive, and Text Analytics for Airline Customer Reviews: A Data-Driven Approach to Customer Recommendation Prediction
