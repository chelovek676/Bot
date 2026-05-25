import pandas as pd
import spacy
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("dataset.csv")
texts = data["text"].tolist()
labels = data["intent"].tolist()

print("Загрузка модели")
nlp = spacy.load("ru_core_news_md")  

def vectorize(text):

    doc = nlp(text)

    vectors = [token.vector for token in doc if token.has_vector and token.vector_norm != 0]
    if vectors:
        return np.mean(vectors, axis=0)
    else:

        return np.zeros(nlp.vocab.vectors_length)

print("Векторизация текстов")
X = np.array([vectorize(text) for text in texts])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Обучение")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced",
    solver="lbfgs",
    multi_class="auto"
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nMetrics:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

joblib.dump(model, "intent_model_embeddings.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(nlp, "nlp_model_embeddings.pkl")

print("   - intent_model_embeddings.pkl")
print("   - label_encoder.pkl") 
print("   - nlp_model_embeddings.pkl")