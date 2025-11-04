import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Descargar dataset público (UCI SMS Spam Collection)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Convertir etiquetas
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenamiento modelo Naive Bayes
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
