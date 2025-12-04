import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from stop_words import get_stop_words
import joblib
import os
# from datasets import load_dataset


# Crear carpeta si no existe
os.makedirs('app/model', exist_ok=True)
# dataset = load_dataset("sms_spam")
# df = pd.DataFrame(dataset["train"])
# df = df.rename(columns={"sms": "message"})
# Cargar dataset local
df = pd.read_csv("app/model/dataset_spam.csv")

# Limpiar etiquetas
df['label'] = df['label'].astype(str).str.lower().str.strip()

# Convertir etiquetas
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Eliminar filas con NaN en la etiqueta
df = df.dropna(subset=['label_num'])


# Convertir etiquetas
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# Vectorización
vectorizer = TfidfVectorizer(stop_words=get_stop_words("spanish"))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar modelo
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluación
pred = model.predict(X_test_tfidf)
print("Precisión:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Guardar modelo y vectorizador
joblib.dump(model, 'app/model/spam_model.pkl')
joblib.dump(vectorizer, 'app/model/vectorizer.pkl')

print("Entrenamiento terminado")
