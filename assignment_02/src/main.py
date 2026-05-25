# DRAW SPECIFICATIONS
# MODEL: Random Forest
# FEATURE EXTRACTION METHOD: Topic Modelling

import os
import math
import argparse

import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
)


# =========================
# CLI
# =========================
def parse_arguments():
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        prog="Language Analytics Assignment 2",
        description=(
            "REAL/FAKE news classification using CountVectorizer + LDA + Random Forest"
        ),
    )

    parser.add_argument(
        "--dataset",
        default=os.path.join("..", "data", "fake_or_real_news.csv"),
        help="Path to CSV dataset."
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "output"),
        help="Directory for outputs."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model and save the best pipeline."
    )
    parser.add_argument(
        "--do_test_eval",
        action="store_true",
        help="Evaluate the saved model on the provided dataset."
    )
    parser.add_argument(
        "--extract_topics",
        action="store_true",
        help="Extract topics and run post-hoc topic analysis from the saved model."
    )

    parser.add_argument(
        "--n_top_words",
        type=int,
        default=10,
        help="Number of top words to show/save per topic."
    )
    parser.add_argument(
        "--n_topic_label_words",
        type=int,
        default=3,
        help="Number of top words to use in mirrored topic plot labels."
    )

    return parser.parse_args()


# =========================
# DATA
# =========================
def load_and_preprocess_dataset(dataset_path, text_col="text", label_col="label"):
    """
    Load dataset and return X (texts) and y (labels).
    """
    data = pd.read_csv(dataset_path)

    if text_col not in data.columns:
        if "text" in data.columns:
            text_col = "text"
        elif "title" in data.columns:
            text_col = "title"
        else:
            raise ValueError("No valid text column found. Expected 'text' or 'title'.")

    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    X = data[text_col].astype(str)
    y = data[label_col].astype(str)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Number of documents: {len(X)}")
    print("Label distribution:")
    print(y.value_counts())

    return X, y


# =========================
# MODEL HELPERS
# =========================
def get_model_path(outdir_path):
    return os.path.join(outdir_path, "best_pipeline.joblib")


def load_saved_pipeline(outdir_path):
    """
    Load saved pipeline from disk.
    """
    model_path = get_model_path(outdir_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Run with --train first."
        )
    return load(model_path)


def get_topic_components_from_pipeline(pipeline):
    """
    Return fitted vectorizer, fitted LDA model, and feature names.
    """
    vect = pipeline.named_steps["vect"]
    lda = pipeline.named_steps["topic"]
    feature_names = vect.get_feature_names_out()
    return vect, lda, feature_names


def transform_texts_to_topic_distributions(pipeline, X_texts):
    """
    Transform texts into document-topic distributions using saved pipeline.
    """
    vect = pipeline.named_steps["vect"]
    lda = pipeline.named_steps["topic"]

    X_bow = vect.transform(X_texts)
    topic_distributions = lda.transform(X_bow)
    return topic_distributions


# =========================
# TRAINING
# =========================
def train_model(X_train, y_train, outdir_path):
    """
    Train GridSearchCV pipeline and save best fitted pipeline.
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(
            lowercase=True,
            min_df=5,
            max_df=0.95,
            stop_words="english"
        )),
        ("topic", LatentDirichletAllocation(random_state=42)),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "vect__max_features": [250, 500, 1000],
        "topic__n_components": [10, 20],
        "topic__learning_decay": [0.25, 0.5, 0.7],
        "clf__n_estimators": [50, 100, 150],
        "clf__max_depth": [None, 20],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid.best_params_)
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

    best_model_pipeline = grid.best_estimator_

    train_preds = best_model_pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds, pos_label="REAL")
    train_report = classification_report(y_train, train_preds)

    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Training F1-score (REAL positive): {train_f1:.4f}")
    print("\nTraining classification report:\n")
    print(train_report)

    with open(os.path.join(outdir_path, "best_train_params.txt"), "w", encoding="utf-8") as f:
        f.write(str(grid.best_params_))

    with open(os.path.join(outdir_path, "best_cv_score.txt"), "w", encoding="utf-8") as f:
        f.write(f"{grid.best_score_}\n")

    with open(os.path.join(outdir_path, "train_accuracy.txt"), "w", encoding="utf-8") as f:
        f.write(f"Training accuracy: {train_acc}\n")

    with open(os.path.join(outdir_path, "train_f1.txt"), "w", encoding="utf-8") as f:
        f.write(f"Training F1-score (REAL positive): {train_f1}\n")

    with open(os.path.join(outdir_path, "train_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(train_report)

    model_path = get_model_path(outdir_path)
    dump(best_model_pipeline, model_path)
    print(f"\nSaved best pipeline to: {model_path}")

    return best_model_pipeline


# =========================
# EVALUATION
# =========================
def evaluate_model_on_test(X_test, y_test, outdir_path):
    """
    Load saved model and evaluate on supplied dataset.
    """
    pipeline = load_saved_pipeline(outdir_path)

    preds = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, preds)
    test_f1 = f1_score(y_test, preds, pos_label="REAL")
    test_report = classification_report(y_test, preds)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test F1-score (REAL positive): {test_f1:.4f}")
    print("\nTest classification report:\n")
    print(test_report)

    class_index = list(pipeline.classes_).index("REAL")
    y_score = pipeline.predict_proba(X_test)[:, class_index]

    test_roc_auc = roc_auc_score(y_test, y_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label="REAL")

    print(f"Test ROC AUC: {test_roc_auc:.4f}")

    with open(os.path.join(outdir_path, "test_accuracy.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {test_acc}\n")

    with open(os.path.join(outdir_path, "test_f1.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test F1-score (REAL positive): {test_f1}\n")

    with open(os.path.join(outdir_path, "test_roc_auc.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test ROC AUC: {test_roc_auc}\n")

    with open(os.path.join(outdir_path, "test_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds
    })
    roc_df.to_csv(os.path.join(outdir_path, "roc_curve_points.csv"), index=False)

    display = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=test_roc_auc,
        estimator_name="Random Forest + LDA"
    )
    display.plot()
    plt.title("ROC Curve on Evaluation Set")
    plt.savefig(os.path.join(outdir_path, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {os.path.join(outdir_path, 'roc_curve.png')}")

    return test_acc


# =========================
# TOPIC EXPORTS
# =========================
def save_topics(pipeline, outdir_path, n_top_words=10):
    """
    Save top words per topic from fitted pipeline.
    """
    _, lda, feature_names = get_topic_components_from_pipeline(pipeline)

    topics_path = os.path.join(outdir_path, "topics.txt")
    with open(topics_path, "w", encoding="utf-8") as f:
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]

            topic_line = ", ".join(
                f"{word} ({weight:.2f})"
                for word, weight in zip(top_words, top_weights)
            )
            f.write(f"Topic {topic_idx + 1}: {topic_line}\n")

    print(f"Saved topics to: {topics_path}")


def plot_top_words(lda, feature_names, n_top_words, title, outdir_path):
    """
    Plot top words for each LDA topic.
    """
    n_topics = lda.components_.shape[0]
    n_cols = 2
    n_rows = math.ceil(n_topics / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        sharex=False
    )

    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = feature_names[top_indices]
        top_weights = topic[top_indices]

        ax = axes[topic_idx]
        ax.barh(top_words[::-1], top_weights[::-1], height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=9)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    for j in range(topic_idx + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(outdir_path, "lda_topics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved topic-word plot to: {save_path}")


# =========================
# POST-HOC TOPIC ANALYSIS
# =========================
def build_topic_distribution_dataframe(topic_distributions, y_labels):
    """
    Build dataframe from document-topic distributions and labels.
    """
    n_topics = topic_distributions.shape[1]

    df = pd.DataFrame(
        topic_distributions,
        columns=[f"topic_{i+1}" for i in range(n_topics)]
    )
    df["label"] = list(y_labels)
    return df


def save_document_topic_distributions(X_texts, y_labels, pipeline, outdir_path):
    """
    Save document-topic distributions for each document.
    """
    topic_distributions = transform_texts_to_topic_distributions(pipeline, X_texts)
    df = build_topic_distribution_dataframe(topic_distributions, y_labels)

    topic_columns = [c for c in df.columns if c.startswith("topic_")]
    df["dominant_topic"] = df[topic_columns].idxmax(axis=1)
    df["text"] = list(X_texts)

    save_path = os.path.join(outdir_path, "document_topic_distributions.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved document-topic distributions to: {save_path}")


def save_average_topic_distribution_by_label(X_texts, y_labels, pipeline, outdir_path):
    """
    Save mean topic proportions for REAL vs FAKE.
    """
    topic_distributions = transform_texts_to_topic_distributions(pipeline, X_texts)
    df = build_topic_distribution_dataframe(topic_distributions, y_labels)

    mean_by_label = df.groupby("label").mean(numeric_only=True)
    save_path = os.path.join(outdir_path, "average_topic_distribution_by_label.csv")
    mean_by_label.to_csv(save_path)

    print(f"Saved average topic distribution by label to: {save_path}")
    print("\nAverage topic distribution by label:")
    print(mean_by_label)


def save_dominant_topic_counts_by_label(X_texts, y_labels, pipeline, outdir_path):
    """
    Save counts of dominant topic per label.
    """
    topic_distributions = transform_texts_to_topic_distributions(pipeline, X_texts)
    df = build_topic_distribution_dataframe(topic_distributions, y_labels)

    topic_columns = [c for c in df.columns if c.startswith("topic_")]
    df["dominant_topic"] = df[topic_columns].idxmax(axis=1)

    counts = pd.crosstab(df["dominant_topic"], df["label"])
    save_path = os.path.join(outdir_path, "dominant_topic_counts_by_label.csv")
    counts.to_csv(save_path)

    print(f"Saved dominant topic counts by label to: {save_path}")
    print("\nDominant topic counts by label:")
    print(counts)


def plot_average_topic_distribution_by_label(X_texts, y_labels, pipeline, outdir_path):
    """
    Plot average topic proportions for REAL vs FAKE.
    """
    topic_distributions = transform_texts_to_topic_distributions(pipeline, X_texts)
    df = build_topic_distribution_dataframe(topic_distributions, y_labels)

    mean_by_label = df.groupby("label").mean(numeric_only=True).T

    ax = mean_by_label.plot(kind="bar", figsize=(14, 6))
    ax.set_title("Average Topic Distribution by Label")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Average Topic Proportion")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(outdir_path, "average_topic_distribution_by_label.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved average topic distribution plot to: {save_path}")


def plot_mirrored_topic_distribution_by_label(
    X_texts,
    y_labels,
    pipeline,
    outdir_path,
    n_top_words_label=3
):
    """
    Create mirrored plot of average topic prevalence in FAKE vs REAL.
    Left side = FAKE (negative values), right side = REAL (positive values).
    Topic labels are defined by top words from the global LDA topics.
    """
    _, lda, feature_names = get_topic_components_from_pipeline(pipeline)

    topic_distributions = transform_texts_to_topic_distributions(pipeline, X_texts)
    df = build_topic_distribution_dataframe(topic_distributions, y_labels)

    mean_by_label = df.groupby("label").mean(numeric_only=True)

    if "FAKE" not in mean_by_label.index or "REAL" not in mean_by_label.index:
        raise ValueError("Both FAKE and REAL must be present in y_labels.")

    fake_means = mean_by_label.loc["FAKE"].values
    real_means = mean_by_label.loc["REAL"].values
    n_topics = len(fake_means)

    topic_labels = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_top_words_label:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_labels.append(f"T{topic_idx + 1}: " + ", ".join(top_words))

    y_pos = list(range(n_topics))

    plt.figure(figsize=(12, max(6, n_topics * 0.45)))
    plt.barh(y_pos, -fake_means, label="FAKE")
    plt.barh(y_pos, real_means, label="REAL")
    plt.yticks(y_pos, topic_labels)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Average Topic Proportion")
    plt.title("Mirrored Average Topic Prevalence: FAKE vs REAL")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(outdir_path, "mirrored_topic_distribution_by_label.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved mirrored topic distribution plot to: {save_path}")


# =========================
# TOPIC EXTRACTION DRIVER
# =========================
def extract_and_plot_topics_from_saved_model(
    X_texts,
    y_labels,
    outdir_path,
    n_top_words=10,
    n_topic_label_words=3
):
    """
    Load saved fitted pipeline, save/plot topics, and run post-hoc topic analysis.
    """
    pipeline = load_saved_pipeline(outdir_path)
    _, lda, feature_names = get_topic_components_from_pipeline(pipeline)

    save_topics(pipeline, outdir_path, n_top_words=n_top_words)

    plot_top_words(
        lda=lda,
        feature_names=feature_names,
        n_top_words=n_top_words,
        title="Top Words per LDA Topic",
        outdir_path=outdir_path
    )

    save_document_topic_distributions(X_texts, y_labels, pipeline, outdir_path)
    save_average_topic_distribution_by_label(X_texts, y_labels, pipeline, outdir_path)
    save_dominant_topic_counts_by_label(X_texts, y_labels, pipeline, outdir_path)
    plot_average_topic_distribution_by_label(X_texts, y_labels, pipeline, outdir_path)
    plot_mirrored_topic_distribution_by_label(
        X_texts,
        y_labels,
        pipeline,
        outdir_path,
        n_top_words_label=n_topic_label_words
    )


# =========================
# MAIN
# =========================
def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.train:
        X, y = load_and_preprocess_dataset(args.dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        train_model(X_train, y_train, args.output_dir)

        print("\nTraining complete.")
        print("You can now run:")
        print("  --do_test_eval")
        print("  --extract_topics")

    if args.do_test_eval:
        X_eval, y_eval = load_and_preprocess_dataset(args.dataset)
        evaluate_model_on_test(X_eval, y_eval, args.output_dir)

    if args.extract_topics:
        X_topics, y_topics = load_and_preprocess_dataset(args.dataset)
        extract_and_plot_topics_from_saved_model(
            X_topics,
            y_topics,
            args.output_dir,
            n_top_words=args.n_top_words,
            n_topic_label_words=args.n_topic_label_words
        )

    if not args.train and not args.do_test_eval and not args.extract_topics:
        print("No action selected.")
        print("Use one or more of:")
        print("  --train")
        print("  --do_test_eval")
        print("  --extract_topics")


if __name__ == "__main__":
    main()