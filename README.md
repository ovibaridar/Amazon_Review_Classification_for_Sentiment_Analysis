# Amazon Review Classification for Sentiment Analysis

Classifies Amazon food reviews as **positive** or **negative** using natural-language processing and classic machine-learning models.

## Workflow

1. **Load and explore:** review scores, class balance, word clouds of positive and negative reviews
2. **Clean the text:** lowercasing, punctuation and stop-word removal with NLTK
3. **Vectorise:** TF-IDF features
4. **Balance the classes:** random oversampling (imbalanced-learn)
5. **Train and compare models:**
   - Multinomial Naive Bayes
   - Bernoulli Naive Bayes
   - Logistic Regression
6. **Evaluate:** train and test accuracy, confusion matrices

## Notebooks

| File | Description |
|---|---|
| `Amazon Review Classification for Sentiment Analysis.ipynb` | Main notebook |
| `amazon-fine-food-reviews.ipynb` | Same pipeline on the full Kaggle *Amazon Fine Food Reviews* dataset |

Dataset: [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## How to run

```bash
git clone https://github.com/ovibaridar/Amazon_Review_Classification_for_Sentiment_Analysis.git
cd Amazon_Review_Classification_for_Sentiment_Analysis
pip install numpy pandas matplotlib seaborn plotly scikit-learn imbalanced-learn nltk wordcloud
jupyter notebook
```

The main notebook loads the data directly from [`Data_sets/amazon.csv`](https://github.com/ovibaridar/Data_sets), so no download is needed. For `amazon-fine-food-reviews.ipynb`, download the Kaggle dataset and save it as `amazon food Reviews.csv` next to the notebook.

**Stack:** Python · NLTK · scikit-learn · imbalanced-learn · pandas · Plotly · WordCloud
