#importando bibliotecas necessárias

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#importando o dataset
st.title('Faça uma análise descritiva das variáveis do escopo')
st.write('Verifique a distribuição dessas variáveis')
st.write('Veja se há valores missing e caso haja, decida o que fazer')
st.write('Faça mais algum tratamento nas variáveis caso ache pertinente')


df = pd.read_csv('online_shoppers_intention.csv', )
df_copia = df.copy()
st.dataframe(df.head().astype(str))



# Análise Descritiva
st.header('Análise Descritiva dos Dados')
st.write(df.describe().transpose())
st.write('Valores Nulos por Coluna:')
st.write(df.isnull().sum())

if len(df) > 5000:
    df_sample = df.sample(5000, random_state=42)
else:
    df_sample = df.copy()

st.write(f"Dataset com {df.shape[0]} linhas e {df.shape[1]} colunas.")

# --- Estatísticas gerais ---
st.subheader("🔍 Estatísticas Gerais")
st.dataframe(df.describe(include='all').transpose())

# --- Seleção de variável ---
st.subheader("📈 Gráficos de Análise Descritiva")

coluna = st.selectbox("Escolha uma variável para visualizar:", df.columns)

# Detectar tipo da variável
if df[coluna].dtype in ['int64', 'float64']:
    # Numérica
    fig, ax = plt.subplots()
    sns.histplot(df_sample[coluna], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribuição de {coluna}")
    st.pyplot(fig)

    # Boxplot para ver outliers
    fig, ax = plt.subplots()
    sns.boxplot(x=df_sample[coluna], ax=ax)
    ax.set_title(f"Boxplot de {coluna}")
    st.pyplot(fig)

elif df[coluna].dtype == 'bool' or df[coluna].nunique() <= 10:
    # Booleana ou categórica com poucas categorias
    fig, ax = plt.subplots()
    df_sample[coluna].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Frequência das categorias de {coluna}")
    st.pyplot(fig)

else:
    # Categórica com muitas categorias
    top_n = st.slider("Número de categorias para exibir:", 5, 20, 10)
    fig, ax = plt.subplots()
    df_sample[coluna].value_counts().head(top_n).plot(kind='bar', ax=ax)
    ax.set_title(f"As {top_n} categorias mais frequentes de {coluna}")
    st.pyplot(fig)

# --- Correlação entre variáveis numéricas ---
st.subheader("📉 Mapa de Correlação (variáveis numéricas)")
corr = df.select_dtypes(include='number').corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, cmap="coolwarm", annot=False)
st.pyplot(fig)


st.title('Número de grupos')
st.write('Utilize as técnicas vistas em aula que te ajudem a decidir pelo número de grupos a ser utilizados.')

kmeans = KMeans(n_clusters=3, max_iter=100, algorithm= 'lloyd', random_state=42)
kmeans.fit(df.select_dtypes(include=['int64', 'float64']))

#identificar grupos no streamlit
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df.select_dtypes(include=['float64', 'int64'])).astype(str)
amostra = df.select_dtypes(include=['float64', 'int64']).copy()
amostra['cluster'] = df['cluster']

n = min(500, len(amostra))
fig = sns.pairplot(amostra.sample(n), hue='cluster', diag_kind='kde')
st.pyplot(fig)

st.title("📊 Avaliação dos Grupos (KMeans)")

st.write("""
Nesta seção, comparamos diferentes quantidades de clusters (k) e analisamos 
como os grupos se diferenciam entre si com base nas variáveis numéricas.
""")

# Seleciona apenas variáveis numéricas
X = df.select_dtypes(include=['float64', 'int64'])

# Normaliza os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Escolher os valores de k que quer comparar ---
valores_k = [3, 4]  # você pode mudar para [2, 3], [3, 5], etc.
for k in valores_k:
    st.subheader(f"🧩 Análise com k = {k}")

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Adiciona os grupos ao dataframe
    df[f'grupo_{k}'] = labels

    # --- Estatísticas descritivas dos grupos ---
    st.write(f"📈 Médias das variáveis numéricas por grupo (k={k})")
    medias = df.groupby(f'grupo_{k}')[X.columns].mean().round(2)
    st.dataframe(medias)

    # --- Gráfico comparativo (exemplo: 'Administrative_Duration' vs 'BounceRates') ---
    colunas_numericas = list(X.columns)
    if len(colunas_numericas) >= 2:
        xcol = colunas_numericas[0]
        ycol = colunas_numericas[1]
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=xcol, y=ycol, hue=f'grupo_{k}', 
            data=df.sample(min(1000, len(df))), 
            palette='tab10', ax=ax
        )
        ax.set_title(f"Distribuição dos clusters (k={k}) - {xcol} x {ycol}")
        st.pyplot(fig)

st.write("""
Com k=3, observam-se três perfis distintos:
- Grupo 0: visitantes de curta duração, baixo engajamento.
- Grupo 1: usuários intermediários, com navegação média.
- Grupo 2: visitantes de alto valor, com alto *PageValues* e *Administrative_Duration*.

Com k=4, a separação é mais sutil e menos interpretável.
Portanto, **k=3 é o agrupamento final escolhido**, representando bem o comportamento dos usuários.
""")

st.header("Avaliação dos Resultados dos Grupos")

# Seleciona o agrupamento final (por exemplo, k=3)
grupo_final = 'grupo_3'

# Garante que existe a coluna do agrupamento escolhido
if grupo_final not in df.columns:
    st.warning(f"O agrupamento '{grupo_final}' não foi gerado. Execute a célula de clusterização com k=3 primeiro.")
else:
    # Avaliação por grupo
    analise = df.groupby(grupo_final).agg({
        'BounceRates': 'mean',
        'Revenue': lambda x: x.mean() * 100  # porcentagem de compradores
    }).round(2)

    analise.rename(columns={'Revenue': 'Taxa de Compra (%)'}, inplace=True)

    st.subheader("📊 Comparativo dos Grupos")
    st.dataframe(analise)

    # Gráfico: taxa de compra por grupo
    fig, ax = plt.subplots()
    sns.barplot(x=analise.index, y='Taxa de Compra (%)', data=analise, palette='viridis', ax=ax)
    ax.set_title("Taxa de Compra (Revenue) por Grupo")
    st.pyplot(fig)

    # Gráfico: BounceRates médio por grupo
    fig, ax = plt.subplots()
    sns.barplot(x=analise.index, y='BounceRates', data=analise, palette='coolwarm', ax=ax)
    ax.set_title("Bounce Rate médio por Grupo")
    st.pyplot(fig)

    # Interpretação automática
    grupo_com_maior_compra = analise['Taxa de Compra (%)'].idxmax()
    taxa_max = analise.loc[grupo_com_maior_compra, 'Taxa de Compra (%)']

    st.subheader("💡 Interpretação")
    st.write(f"""
    O grupo com **maior propensão à compra** é o **Grupo {grupo_com_maior_compra}**, 
    com uma **taxa média de compra de {taxa_max:.1f}%**.

    Além disso, ao comparar as médias de *BounceRates*, 
    é possível observar que grupos com **menor taxa de rejeição** (BounceRates) 
    tendem a apresentar **maior taxa de conversão (Revenue)**.
    """)