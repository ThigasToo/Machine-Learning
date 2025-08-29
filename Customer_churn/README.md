# 📊 Customer Churn Prediction

Este projeto implementa um sistema de **predição de churn de clientes** (saída/cancelamento de serviço) utilizando **Machine Learning**.  
O objetivo é prever se um cliente irá cancelar o serviço com base em dados históricos de clientes da **Telco Company**.

---

## 📌 Funcionalidades

- **Exploração e análise dos dados (EDA)** com histogramas, boxplots e heatmaps.
- **Tratamento de dados**:
  - Remoção de colunas irrelevantes.
  - Conversão de variáveis categóricas em numéricas via **Label Encoding**.
  - Tratamento de valores faltantes e inconsistentes.
- **Balanceamento da base de dados** com **SMOTE** para lidar com desbalanceamento de classes.
- **Treinamento de modelos**:
  - Decision Tree
  - Random Forest
  - XGBoost (XGBRFClassifier)
- **Validação cruzada (cross-validation)** para avaliação de desempenho.
- **Exportação do modelo treinado** e dos **encoders** em arquivos `.pkl`.
- **Sistema de predição** que carrega o modelo salvo e permite realizar previsões para novos clientes.

---

## 🚀 Tecnologias Utilizadas

- **Linguagem**: Python 3
- **Bibliotecas principais**:
  - [pandas](https://pandas.pydata.org/) → manipulação de dados
  - [numpy](https://numpy.org/) → cálculos numéricos
  - [matplotlib](https://matplotlib.org/) e [seaborn](https://seaborn.pydata.org/) → visualizações
  - [scikit-learn](https://scikit-learn.org/) → modelos de ML e pré-processamento
  - [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/stable/) → balanceamento de classes
  - [xgboost](https://xgboost.readthedocs.io/) → modelo XGBoost
  - [pickle](https://docs.python.org/3/library/pickle.html) → salvar/carregar modelo e encoders

---

## 📊 Fluxo do Projeto

1. **Carregamento e análise do dataset**
   - Visualização de colunas, valores únicos e estatísticas descritivas.

2. **Pré-processamento**
   - Conversão da coluna `TotalCharges` para float.
   - Substituição de valores faltantes.
   - Label Encoding de variáveis categóricas.

3. **Balanceamento**
   - Aplicação do **SMOTE** nos dados de treino.

4. **Treinamento**
   - Teste com três modelos:
     - Decision Tree
     - Random Forest
     - XGBoost
   - Melhor desempenho obtido com **Random Forest**.

5. **Avaliação**
   - Acurácia
   - Matriz de confusão
   - Relatório de classificação (precision, recall, f1-score)

6. **Exportação**
   - Modelo salvo em `customer_churn_model.pkl`.
   - Encoders salvos em `encoders.pkl`.

7. **Predição**
   - Exemplo de entrada de dados para novos clientes.
   - Transformação dos dados com encoders salvos.
   - Predição do churn com probabilidades.

---

## 📈 Resultados Obtidos

- Melhor modelo: **Random Forest**
- Boa performance em termos de **acurácia e balanceamento de classes**.
- Sistema pronto para realizar previsões com novos dados.

---

## 🔮 Exemplo de Uso

```python
import pickle
import pandas as pd

# Carregar modelo salvo
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
loaded_features_names = model_data["features_names"]

# Carregar encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Criar novo cliente para predição
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.35,
    'TotalCharges': 350.5
}

# Transformar em DataFrame
input_df = pd.DataFrame([input_data])

# Aplicar encoders
for col, encoder in encoders.items():
    if col in input_df.columns:
        input_df[col] = encoder.transform(input_df[col])

# Fazer predição
prediction = loaded_model.predict(input_df)
prediction_prob = loaded_model.predict_proba(input_df)

print("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
print("Probability:", prediction_prob)
