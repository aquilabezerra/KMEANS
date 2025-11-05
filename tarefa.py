#importando bibliotecas necessárias

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# 🔹 Leitura e Exibição Inicial
# ==============================
st.title('Análise Descritiva e Clusterização - Online Shoppers Intention')

df = pd.read_csv('online_shoppers_intention.csv')
df_copia = df.copy()

st.subheader('Visualização inicial do dataset')
st.dataframe(df.head().astype(str))
st.write(f"**Shape do dataset:** {df.shape[0]} linhas × {df.shape[1]} colunas")

# ==============================
# 🔹 Análise Descritiva
# ==============================
st.header('Análise Descritiva')
st.write('Verifique a distribuição das variáveis e se há valores ausentes.')

st.write('**Valores Nulos por Coluna:**')
st.write(df.isnull().sum())

st.subheader("Estatísticas Gerais")
st.dataframe(df.describe(include='all').transpose())

# ==============================
# 🔹 Visualização interativa
# ==============================
st.subheader("Gráficos de Análise Descritiva")

if len(df) > 5000:
    df_sample = df.sample(5000, random_state=42)
else:
    df_sample = df.copy()

coluna = st.selectbox("Escolha uma variável para visualizar:", df.columns)

if df[coluna].dtype in ['int64', 'float64']:
    fig, ax = plt.subplots()
    sns.histplot(df_sample[coluna], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribuição de {coluna}")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x=df_sample[coluna], ax=ax)
    ax.set_title(f"Boxplot de {coluna}")
    st.pyplot(fig)

elif df[coluna].dtype == 'bool' or df[coluna].nunique() <= 10:
    fig, ax = plt.subplots()
    df_sample[coluna].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Frequência das categorias de {coluna}")
    st.pyplot(fig)

else:
    top_n = st.slider("Número de categorias para exibir:", 5, 20, 10)
    fig, ax = plt.subplots()
    df_sample[coluna].value_counts().head(top_n).plot(kind='bar', ax=ax)
    ax.set_title(f"As {top_n} categorias mais frequentes de {coluna}")
    st.pyplot(fig)

# ==============================
# 🔹 Correlação
# ==============================
st.subheader(" Mapa de Correlação (variáveis numéricas)")
corr = df.select_dtypes(include='number').corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, cmap="coolwarm", annot=False)
st.pyplot(fig)

# ==============================
# 🔹 Determinação do número de grupos
# ==============================
st.header(' Escolha do número de grupos (Método do Cotovelo)')

X = df.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, algorithm='lloyd')
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(k_values, inertias, marker='o')
ax.set_title("Método do Cotovelo (Elbow Method)")
ax.set_xlabel("Número de clusters (k)")
ax.set_ylabel("Inércia (WCSS)")
st.pyplot(fig)

st.write("""
🔹 **Interpretação:**  
Escolha o valor de *k* onde a curva começa a se estabilizar — esse é o “cotovelo” e indica o número ideal de grupos.
""")

# ==============================
# 🔹 Clusterização e Avaliação
# ==============================
st.header("Avaliação dos Agrupamentos")

k_escolhido = st.slider("Escolha o número de clusters (k):", 2, 10, 3)

kmeans = KMeans(n_clusters=k_escolhido, random_state=42, algorithm='lloyd')
labels = kmeans.fit_predict(X_scaled)
df[f'cluster_{k_escolhido}'] = labels

st.write(f"Médias das variáveis numéricas por grupo (k={k_escolhido})")
medias = df.groupby(f'cluster_{k_escolhido}')[X.columns].mean().round(2)
st.dataframe(medias)

colunas_numericas = list(X.columns)
if len(colunas_numericas) >= 2:
    xcol = st.selectbox("Eixo X do gráfico:", colunas_numericas, index=0)
    ycol = st.selectbox("Eixo Y do gráfico:", colunas_numericas, index=1)
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=xcol, y=ycol, hue=f'cluster_{k_escolhido}',
        data=df.sample(min(1000, len(df))), palette='tab10', ax=ax
    )
    ax.set_title(f"Distribuição dos clusters (k={k_escolhido})")
    st.pyplot(fig)

# ==============================
# 🔹 Avaliação dos Resultados
# ==============================
st.header("Avaliação dos Resultados (Variáveis fora do escopo)")

if 'Revenue' in df.columns and 'BounceRates' in df.columns:
    analise = df.groupby(f'cluster_{k_escolhido}').agg({
        'BounceRates': 'mean',
        'Revenue': lambda x: x.mean() * 100
    }).round(2)
    analise.rename(columns={'Revenue': 'Taxa de Compra (%)'}, inplace=True)

    st.subheader("Comparativo dos Grupos")
    st.dataframe(analise)

    fig, ax = plt.subplots()
    sns.barplot(x=analise.index, y='Taxa de Compra (%)',
                data=analise, palette='viridis', ax=ax)
    ax.set_title("Taxa de Compra (Revenue) por Grupo")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(x=analise.index, y='BounceRates',
                data=analise, palette='coolwarm', ax=ax)
    ax.set_title("Bounce Rate médio por Grupo")
    st.pyplot(fig)

    grupo_top = analise['Taxa de Compra (%)'].idxmax()
    st.success(
        f"O grupo mais propenso à compra é o **Cluster {grupo_top}**, com taxa média de compra de {analise.loc[grupo_top, 'Taxa de Compra (%)']}%.")

# ==============================
# 🔹 Renomear os Clusters
# ==============================
st.header("Renomear os Clusters")

st.write("Dê nomes aos clusters com base nas suas características observadas:")

nomes_clusters = {}
for i in sorted(df[f'cluster_{k_escolhido}'].unique()):
    nome = st.text_input(f"Nome para o Cluster {i}:", f"Grupo {i}")
    nomes_clusters[i] = nome

df['Nome_Cluster'] = df[f'cluster_{k_escolhido}'].map(nomes_clusters)

st.write("### Clusters com novos nomes:")
st.dataframe(df[['Nome_Cluster', 'Revenue', 'BounceRates']].head())

st.write("""
**Exemplo de nomes:**
- Cluster 0 → Visitantes Casuais  
- Cluster 1 → Pesquisadores  
- Cluster 2 → Clientes Engajados
""")
