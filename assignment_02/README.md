# Language Analytics Assignment 2

## REAL/FAKE news classification with topic modelling and Random Forests

This repository contains the code and outputs for Assignment 2 in Language Analytics. The assignment investigates whether REAL and FAKE news articles can be distinguished using **topic modelling** as the feature representation method and a **Random Forest classifier** as the classification model.

The project uses a scikit-learn pipeline consisting of:

1. `CountVectorizer` for converting article text into word-count features.
2. `LatentDirichletAllocation` (LDA) for reducing the document-term matrix into topic distributions.
3. `RandomForestClassifier` for classifying articles as `REAL` or `FAKE` based on their topic distributions.

The main goal is not only to achieve strong classification performance, but also to evaluate whether the topics learned by LDA reveal meaningful thematic differences between REAL and FAKE articles.

---

## Repository structure

```text
assignment_02/
├── data/
│   ├── fake_real_news_train_data.csv
│   └── TOP_SECRET_fake_real_news_test_data.csv
├── output/
│   ├── best_pipeline.joblib
│   ├── best_train_params.txt
│   ├── test_accuracy.txt
│   ├── test_f1.txt
│   ├── test_roc_auc.txt
│   ├── test_classification_report.txt
│   ├── roc_curve.png
│   ├── lda_topics.png
│   ├── topics.txt
│   ├── average_topic_distribution_by_label.csv
│   ├── average_topic_distribution_by_label.png
│   ├── mirrored_topic_distribution_by_label.png
│   ├── dominant_topic_counts_by_label.csv
│   └── document_topic_distributions.csv
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```

---

## Data

The project uses a binary REAL/FAKE news dataset provided for the course. The data is split into:

- `data/fake_real_news_train_data.csv`: training data used for fitting and hyperparameter search.
- `data/TOP_SECRET_fake_real_news_test_data.csv`: held-out test data used for final evaluation.

Each dataset contains article text and a binary label:

- `REAL`: genuine news article.
- `FAKE`: article belonging to the fake-news category in the original dataset.

The written report discusses the dataset source and the limitations of treating the labels as simple binary ground truth.

---

## Setup

From the root of the project folder, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The project was run with the package versions listed in `requirements.txt`.

---

## How to run the analysis

All commands should be run from the root folder, `assignment_02/`.

### 1. Train the model

```bash
python src/main.py \
  --train \
  --dataset data/fake_real_news_train_data.csv \
  --output_dir output
```

This trains the full scikit-learn pipeline using grid search and saves the best model to:

```text
output/best_pipeline.joblib
```

The best hyperparameters are saved in:

```text
output/best_train_params.txt
```

### 2. Evaluate the saved model on the test set

```bash
python src/main.py \
  --do_test_eval \
  --dataset data/TOP_SECRET_fake_real_news_test_data.csv \
  --output_dir output
```

This loads the saved model and evaluates it on the held-out test data. It creates:

```text
output/test_accuracy.txt
output/test_f1.txt
output/test_roc_auc.txt
output/test_classification_report.txt
output/roc_curve.png
output/roc_curve_points.csv
```

### 3. Extract and visualize topics

```bash
python src/main.py \
  --extract_topics \
  --dataset data/TOP_SECRET_fake_real_news_test_data.csv \
  --output_dir output
```

This loads the saved pipeline, extracts the LDA topics, transforms the supplied articles into document-topic distributions, and creates the topic visualizations used in the report.

Important output files include:

```text
output/topics.txt
output/lda_topics.png
output/average_topic_distribution_by_label.csv
output/average_topic_distribution_by_label.png
output/mirrored_topic_distribution_by_label.png
output/dominant_topic_counts_by_label.csv
output/document_topic_distributions.csv
```

---

## Reproduced model performance

The saved model achieved the following final test-set performance:

```text
Accuracy: 0.9692
F1-score for REAL: 0.9690
ROC AUC: 0.9929
```

The best hyperparameter configuration was:

```python
{
    "vect__max_features": 1000,
    "topic__n_components": 20,
    "topic__learning_decay": 0.25,
    "clf__n_estimators": 100,
    "clf__max_depth": None,
}
```

The final classification report is saved in:

```text
output/test_classification_report.txt
```

---

## Interpretation of outputs

The most important visual outputs are:

- `roc_curve.png`: shows test-set classification performance across decision thresholds.
- `lda_topics.png`: shows the top words for each LDA topic.
- `mirrored_topic_distribution_by_label.png`: compares the average topic prevalence in FAKE and REAL articles.

The mirrored topic plot is especially important for the written report because it shows that some LDA topics are more common in one class than the other. However, these topics should be interpreted carefully. LDA is unsupervised, so it does not learn topics specifically designed to separate REAL and FAKE articles. Instead, the class comparison is performed after the topic model has already been fitted.

---

## Notes on reproducibility and limitations

The model uses `random_state=42` for both LDA and Random Forests to make the results more reproducible. However, exact results can still vary slightly across package versions or operating systems.

The model performs very well on the provided test set, but this should not be interpreted as proof that it can detect misinformation generally. The learned topics reflect the specific dataset used here, including its U.S. political focus and its particular news period. The model may therefore learn dataset-specific thematic patterns rather than a universal distinction between truth and misinformation.

The written report discusses these limitations in more detail, including dataset bias, inherited labels, topic interpretability, and the limited hyperparameter search space.
