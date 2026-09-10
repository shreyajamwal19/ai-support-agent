"""Baseline A (trivial) and Baseline B (meaningful) intent classifiers, per assignment
Phase 10. Documented exactly what each does -- no strawman baselines.
"""
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_DIR = Path("artifacts/models")


class MajorityClassBaseline:
    """Baseline A (trivial): always predicts the single most frequent intent observed in
    training data. Establishes the floor -- any real system must beat this by a wide
    margin on macro-F1 (majority-class baselines can look deceptively strong on accuracy
    alone when one class dominates, which is exactly why we also report macro-F1)."""
    def __init__(self):
        self.majority = None

    def fit(self, intents: pd.Series):
        self.majority = intents.value_counts().idxmax()
        return self

    def predict(self, texts):
        return [self.majority] * len(texts)


class TfidfLogRegBaseline:
    """Baseline B (meaningful): TF-IDF features + multinomial Logistic Regression, trained
    on rule-classifier-derived weak/silver labels over `baseline_eval_pool_en` (NOT the
    golden set -- no leakage). This is a real, commonly-used, cheap-to-train approach that
    a competent engineer would reach for before an LLM. Because its training labels are
    themselves silver (rule-derived, not human), we do not expect it to exceed the rules it
    was trained to imitate on non-golden data -- its value is (a) generalizing rule
    coverage to phrasing the rules didn't literally match, and (b) providing calibrated-ish
    class probabilities the regex rules can't. See REPORT.md baseline comparison for the
    actual measured numbers and this caveat restated in context.
    """
    def __init__(self):
        self.vectorizer = None
        self.clf = None

    def fit(self, texts, labels):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english",
                                           ngram_range=(1, 2), min_df=3)
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X)

    def predict_proba_top(self, texts):
        X = self.vectorizer.transform(texts)
        probs = self.clf.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.clf.classes_[idx], probs.max(axis=1)

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "tfidf_logreg.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf}, f)

    def load(self):
        with open(MODEL_DIR / "tfidf_logreg.pkl", "rb") as f:
            d = pickle.load(f)
        self.vectorizer, self.clf = d["vectorizer"], d["clf"]
        return self


def train_baseline_b():
    import sys
    sys.path.insert(0, ".")
    from src.intent.rule_classifier import classify
    df = pd.read_parquet("data/processed/baseline_eval_pool_en.parquet")
    prov = df["customer_text_clean"].apply(classify)
    df["silver_label"] = prov.apply(lambda d: d["intent"])
    model = TfidfLogRegBaseline().fit(df["customer_text_clean"], df["silver_label"])
    model.save()
    print("Trained Baseline B (TF-IDF+LogReg) on", len(df), "silver-labeled examples")
    print(df["silver_label"].value_counts())
    return model


if __name__ == "__main__":
    train_baseline_b()
