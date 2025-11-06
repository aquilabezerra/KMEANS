#importando bibliotecas necessárias

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set()


st.set_page_config(page_title="Análise de Cluster - Online Shoppers", page_icon="🛍️", layout="wide")


st.title("Análise de Cluster - Online Shoppers Intention")

df = pd.read_csv("online_shoppers_intention.csv")
st.write(f"**Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.**")

st.dataframe(df.head().astype(str))

st.header("Análise Descritiva dos Dados")
st.write("### Estatísticas gerais:")
st.dataframe(df.describe(include="all").transpose())

st.write("### Valores nulos por coluna:")
st.dataframe(df.isnull().sum())

st.header("Visualização de Variáveis")

coluna = st.selectbox("Escolha uma variável para visualizar:", df.columns)

if df[coluna].dtype in ['int64', 'float64']:
    fig, ax = plt.subplots()
    sns.histplot(df[coluna], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribuição de {coluna}")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x=df[coluna], ax=ax)
    ax.set_title(f"Boxplot de {coluna}")
    st.pyplot(fig)

elif df[coluna].dtype == 'bool' or df[coluna].nunique() <= 10:
    fig, ax = plt.subplots()
    df[coluna].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Frequência das categorias de {coluna}")
    st.pyplot(fig)
else:
    top_n = st.slider("Número de categorias para exibir:", 5, 20, 10)
    fig, ax = plt.subplots()
    df[coluna].value_counts().head(top_n).plot(kind="bar", ax=ax)
    ax.set_title(f"As {top_n} categorias mais frequentes de {coluna}")
    st.pyplot(fig)


st.header("Mapa de Correlação (variáveis numéricas)")
corr = df.select_dtypes(include='number').corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, cmap="coolwarm", annot=False)
st.pyplot(fig)

st.header("Padronização dos Dados")
X = df.select_dtypes(include=['int64', 'float64'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.write("As variáveis numéricas foram padronizadas com **StandardScaler** (média=0, desvio=1).")

st.header("Escolha do Número Ideal de Clusters (Método do Cotovelo)")

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, algorithm="lloyd")
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_xlabel("Número de clusters (k)")
ax.set_ylabel("Inércia (WCSS)")
ax.set_title("Método do Cotovelo")
st.pyplot(fig)

st.info("Observe o ponto onde a curva começa a diminuir lentamente — esse é o 'cotovelo', que indica um bom número de clusters.")

st.header("Comparando soluções de agrupamento")

k_values = [3, 5]
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, algorithm='lloyd')
    df[f'cluster_{k}'] = kmeans.fit_predict(X_scaled)
    st.subheader(f"Agrupamento com k = {k}")
    st.dataframe(df.groupby(f'cluster_{k}')[['BounceRates', 'Revenue']].agg(['mean', 'count']))

st.header("Aplicando K-Means com número de clusters escolhido")

num_clusters = st.slider("Escolha o número de clusters:", 2, 10, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, algorithm="lloyd")
df["cluster_final"] = kmeans.fit_predict(X_scaled)

st.header("Nomeando os Clusters")

cluster_names = {}
for cluster_id in sorted(df["cluster_final"].unique()):
    name = st.text_input(f"Nome para o cluster {cluster_id}:", f"Cluster {cluster_id}")
    cluster_names[cluster_id] = name

df["cluster_nomeado"] = df["cluster_final"].map(cluster_names)

st.write("Visualização com nomes personalizados:")
st.dataframe(df[["cluster_final", "cluster_nomeado", "BounceRates", "Revenue"]].head())

st.header("Avaliação Final dos Clusters")
st.write("Comparando métricas de desempenho por grupo:")

cluster_summary = df.groupby("cluster_nomeado")[["BounceRates", "Revenue"]].agg(["mean", "count"]).sort_values(by=("Revenue", "mean"), ascending=False)
st.dataframe(cluster_summary)

st.success("Grupo com maior média de Revenue representa clientes mais propensos à compra!")
