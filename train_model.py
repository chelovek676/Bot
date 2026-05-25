import pandas as pd
import spacy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("dataset.csv")
texts = data["text"].tolist()
labels = data["intent"].tolist()

nlp = spacy.load("ru_core_news_sm")

def preprocess(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.text.strip():
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)

processed_texts = [preprocess(text) for text in texts]

X_train, X_test, y_train, y_test = train_test_split(
    processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Metrics:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

joblib.dump(pipeline, "intent_model.pkl")
joblib.dump(nlp, "nlp_model.pkl")

print("Model saved to intent_model.pkl")
print("NLP model saved to nlp_model.pkl")