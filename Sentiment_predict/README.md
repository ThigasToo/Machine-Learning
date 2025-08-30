# 🎥 Análise de Sentimentos de Críticas de Filmes com LSTM

Este projeto utiliza uma rede neural recorrente do tipo **LSTM (Long Short-Term Memory) Bidirecional** para realizar análise de sentimentos em críticas de filmes do dataset IMDB. O modelo é treinado para classificar uma crítica como positiva, negativa ou neutra.

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Dataset](#-dataset)
- [Como Funciona](#-como-funciona)
- [Arquitetura do Modelo](#-arquitetura-do-modelo)
- [Requisitos](#-requisitos)
- [Como Executar](#-como-executar)
- [Sistema de Predição](#-sistema-de-predição)

---

## 🚀 Visão Geral

O objetivo é construir um sistema preditivo que, ao receber uma crítica de filme em formato de texto, consiga determinar o sentimento expresso. O projeto abrange todas as etapas do pipeline de um projeto de NLP (Processamento de Linguagem Natural):

1.  **Coleta de Dados:** Utiliza a API do Kaggle para obter o dataset.
2.  **Pré-processamento e Limpeza:** Os textos das críticas são limpos removendo-se tags HTML, pontuações, números e *stopwords* (palavras comuns como "the", "a", "is").
3.  **Tokenização:** Os textos são convertidos em sequências de números para que possam ser processados pela rede neural.
4.  **Construção do Modelo:** Uma rede neural sequencial é construída com camadas de `Embedding`, `LSTM Bidirecional` e `Dense`.
5.  **Treinamento:** O modelo é treinado com os dados de treino e otimizado para minimizar a perda (`binary_crossentropy`). Callbacks como `EarlyStopping` são usados para evitar overfitting.
6.  **Avaliação:** O desempenho do modelo é medido em um conjunto de teste, utilizando métricas como acurácia, `classification_report` e `confusion_matrix`.
7.  **Sistema Preditivo:** Uma função é implementada para classificar novas críticas de forma simples e direta.

## 💾 Dataset

O projeto utiliza o [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Este dataset contém 50.000 críticas de filmes, divididas igualmente em 25.000 para treino e 25.000 para teste, com um balanceamento de 50% de críticas positivas e 50% negativas.

O script assume que o arquivo `IMDB Dataset.csv.zip` está no diretório raiz ou será baixado via API do Kaggle.

## 🛠️ Como Funciona

O fluxo de trabalho do script é o seguinte:

1.  **Configuração da API Kaggle:** O script lê as credenciais de um arquivo `kaggle.json` para se autenticar e interagir com a plataforma Kaggle.
2.  **Carregamento e Divisão dos Dados:** O dataset é carregado com `pandas`, os rótulos ('positive'/'negative') são convertidos para `1` e `0`, e os dados são divididos em 80% para treino e 20% para teste.
3.  **Limpeza de Texto:** Uma função `clean_text` utiliza expressões regulares (`re`) e a biblioteca `NLTK` para pré-processar cada crítica.
4.  **Tokenização e Padding:** O `Tokenizer` do Keras converte as palavras em números (tokens). Todas as sequências são então padronizadas para um comprimento máximo de 200 palavras (`pad_sequences`). Sequências menores são preenchidas com zeros e as maiores são truncadas.
5.  **Treinamento:** O modelo `fit` é chamado nos dados de treino. O `EarlyStopping` monitora a perda de validação (`val_loss`) e interrompe o treinamento se não houver melhora, restaurando os melhores pesos encontrados.
6.  **Avaliação:** Após o treino, `model.evaluate` calcula a perda e a acurácia no conjunto de teste. Um relatório de classificação e uma matriz de confusão são impressos para uma análise mais detalhada.

## 🧠 Arquitetura do Modelo

O modelo é construído usando a API Sequencial do Keras com as seguintes camadas:

1.  **Embedding Layer**: Converte os tokens (números inteiros) em vetores densos de 128 dimensões. Vocabulário máximo de 20.000 palavras.
    - `Embedding(input_dim=20000, output_dim=128, input_length=200)`
2.  **Primeira Camada LSTM Bidirecional**: Processa a sequência em ambas as direções (para frente e para trás) para capturar melhor o contexto. Retorna a sequência completa para a próxima camada.
    - `Bidirectional(LSTM(128, return_sequences=True))`
3.  **Dropout**: Zera aleatoriamente 30% das unidades de entrada para evitar overfitting.
    - `Dropout(0.3)`
4.  **Segunda Camada LSTM Bidirecional**: Uma camada LSTM bidirecional adicional com 64 unidades.
    - `Bidirectional(LSTM(64))`
5.  **Camada Densa (ReLU)**: Uma camada oculta com 64 neurônios e função de ativação ReLU.
    - `Dense(64, activation='relu')`
6.  **Dropout**: Outra camada de dropout de 30%.
    - `Dropout(0.3)`
7.  **Camada de Saída (Sigmoid)**: A camada final com 1 neurônio e ativação `sigmoid`, que gera uma probabilidade entre 0 e 1, ideal para classificação binária.
    - `Dense(1, activation='sigmoid')`

## ⚙️ Requisitos

Para executar este projeto, você precisará das seguintes bibliotecas:

-   `tensorflow`
-   `pandas`
-   `scikit-learn`
-   `nltk`
-   `kaggle`

Você pode instalá-las com pip:
```bash
pip install tensorflow pandas scikit-learn nltk kaggle
```
O script também baixa o corpus de stopwords do NLTK.

## ▶️ Como Executar:
1. Credenciais do Kaggle:
   - Faça o download do seu token de API do Kaggle (arquivo kaggle.json).
   - Coloque o arquivo kaggle.json no mesmo diretório do script.
2. Dataset:
   - O script tentará descompactar o arquivo IMDB Dataset.csv.zip. Certifique-se de que este arquivo esteja no diretório raiz. Você pode baixá-lo manualmente ou usar a API do Kaggle.
3. Execução:
   - Execute o script Python. Ele cuidará de todo o processo de pré-processamento, treinamento e avaliação.
   ```bash
   python sentiment_predict2.py
   ```

## 🎯 Sistema de Predição
O script inclui uma função predict_sentiment(review) que permite classificar novas críticas facilmente. A função classifica o resultado da predição em três categorias:
   - positive: se a probabilidade for maior que 0.7.
   - negative: se a probabilidade for menor que 0.3.
   - neutral: se a probabilidade estiver entre 0.3 e 0.7.
Exemplo de uso:
```python
# Exemplo de uso
new_review = 'This movie was fantastic. I loved every minute of it.'
sentiment = predict_sentiment(new_review)
print(f'The sentiment of the review is: {sentiment}')

# Outro exemplo
new_review = 'The movie was ok, but not good.'
sentiment = predict_sentiment(new_review)
print(f'The sentiment of the review is: {sentiment}')
```
