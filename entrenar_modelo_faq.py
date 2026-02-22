import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Cargar el dataset 

df = pd.read_csv("faq_dataset.csv", sep=";")

X = df["texto"]      # preguntas del usuario
y = df["etiqueta"]   # categoría correcta

# Separar entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Convertir texto a número

vectorizador = CountVectorizer()
X_train_vec = vectorizador.fit_transform(X_train)
X_test_vec = vectorizador.transform(X_test)

# Entrenar modelo supervisado

modelo = MultinomialNB()
modelo.fit(X_train_vec, y_train)

# Evaluación básica

preds = modelo.predict(X_test_vec)
print("ACCURACY:", round(accuracy_score(y_test, preds), 2))
print("\nREPORT:\n", classification_report(y_test, preds))

# Guardar vectorizador y modelo en el pkl

with open("modelo_faq.pkl", "wb") as f:
    pickle.dump((vectorizador, modelo), f)

print("\n Modelo guardado en modelo_faq.pkl")
