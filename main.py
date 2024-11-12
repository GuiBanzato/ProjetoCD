import arxiv
import pandas as pd

# Definindo as categorias para coleta (Exemplo: IA e aprendizado de máquina)
categories = ["cs.AI", "stat.ML"]
num_articles = 400  # Número de artigos por categoria

# Função para coletar artigos de uma categoria
def fetch_arxiv_data(category, num_articles):
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=num_articles,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    data = []
    for result in search.results():
        data.append({
            'title': result.title,
            'summary': result.summary,
            'category': category
        })
    return pd.DataFrame(data)

# Coletando dados de todas as categorias
df_list = [fetch_arxiv_data(cat, num_articles) for cat in categories]
df = pd.concat(df_list, ignore_index=True)

# Salvando os dados em um arquivo CSV (opcional)
df.to_csv("arxiv_data.csv", index=False)
print(f"Coletamos {len(df)} artigos.")
df.head()

import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Limpeza de texto
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
    text = re.sub(r'\W', ' ', text)   # Remove caracteres especiais
    text = text.lower()               # Converte para minúsculas
    return text

# Aplicando limpeza no texto
df['cleaned_summary'] = df['summary'].apply(clean_text)

# Convertendo as classes em números
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# Separando X e y
X = df['cleaned_summary']
y = df['category_encoded']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Pré-processamento concluído. Pronto para a próxima etapa.")

import tensorflow_hub as hub
import numpy as np

# Carregando o Universal Sentence Encoder para embeddings
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Função para converter textos em embeddings
def get_embeddings(texts):
    return np.array([embed([text])[0].numpy() for text in texts])

# Convertendo os textos em embeddings
X_train_embeddings = get_embeddings(X_train)
X_test_embeddings = get_embeddings(X_test)

print("Embeddings criados com sucesso.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Treinando o modelo
clf = RandomForestClassifier()
clf.fit(X_train_embeddings, y_train)

# Avaliando o modelo
y_pred = clf.predict(X_test_embeddings)
accuracy = accuracy_score(y_test, y_pred)

print(f"Acurácia do modelo: {accuracy * 100:.2f}%")


import joblib
joblib.dump(clf, 'modelo_treinado.pkl')
