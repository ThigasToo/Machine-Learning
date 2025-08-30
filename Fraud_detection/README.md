# Detecção de Fraudes em Transações Financeiras

## Fraud_detection(1).py

Este projeto desenvolve um modelo de Machine Learning para detectar transações financeiras fraudulentas. Utilizando um dataset sintético que simula transações de um serviço financeiro móvel, o objetivo é construir um classificador capaz de identificar fraudes com base nas características da transação.

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Dataset](#-dataset)
- [Análise Exploratória de Dados (EDA)](#-análise-exploratória-de-dados-eda)
- [Metodologia](#-metodologia)
- [Requisitos](#-requisitos)
- [Como Executar](#-como-executar)
- [Resultados](#-resultados)
- [Processo de Melhoria do Modelo de Detecção de Fraude (Passos)](#-processo-de-melhoria-do-modelo-de-detecção-de-fraude-passos)

---

## 🚀 Visão Geral

O projeto aborda o problema da detecção de fraudes, que é um desafio clássico devido à alta desproporção entre transações legítimas e fraudulentas (dados desbalanceados).

O fluxo de trabalho consiste em:
1.  **Carregamento e Limpeza:** Leitura do dataset e verificação inicial de consistência e valores ausentes.
2.  **Análise Exploratória (EDA):** Investigação aprofundada dos dados para extrair insights, visualizar distribuições, entender a relação entre as variáveis e identificar padrões de fraude.
3.  **Engenharia de Features:** Criação de novas variáveis (`balanceDiffOrig`, `balanceDiffDest`) para enriquecer o modelo.
4.  **Pré-processamento:** Preparação dos dados para o modelo, incluindo a padronização de features numéricas e a codificação de features categóricas.
5.  **Treinamento do Modelo:** Utilização de um modelo de **Regressão Logística** com ajuste para o desbalanceamento de classes.
6.  **Avaliação:** Medição da performance do modelo com métricas como `classification_report` e `confusion_matrix`.
7.  **Serialização:** Salvamento do pipeline treinado para uso futuro.

## 💾 Dataset

O projeto utiliza o arquivo `AIML Dataset.csv`, um dataset sintético gerado usando o simulador PaySim. Ele contém as seguintes colunas:

-   `step`: Unidade de tempo no mundo real (1 step = 1 hora).
-   `type`: Tipo de transação (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
-   `amount`: Valor da transação.
-   `nameOrig`: Cliente que iniciou a transação.
-   `oldbalanceOrg`: Saldo do remetente antes da transação.
-   `newbalanceOrig`: Saldo do remetente após a transação.
-   `nameDest`: Cliente destinatário da transação.
-   `oldbalanceDest`: Saldo do destinatário antes da transação.
-   `newbalanceDest`: Saldo do destinatário após a transação.
-   `isFraud`: **Variável Alvo**. `1` se a transação for fraudulenta, `0` caso contrário.
-   `isFlaggedFraud`: Sinalizador do sistema para tentativas de transferir valores anormais.

## 📊 Análise Exploratória de Dados (EDA)

Principais descobertas da análise:

-   **Desbalanceamento Extremo:** Apenas **0.13%** das transações no dataset são fraudulentas, o que exige estratégias específicas de modelagem.
-   **Tipos de Transação e Fraude:** Fraudes ocorrem exclusivamente em transações do tipo `TRANSFER` e `CASH_OUT`.
-   **Distribuição de Valores:** A distribuição dos valores de transação (`amount`) é altamente assimétrica, sendo tratada com uma transformação logarítmica para melhor visualização.
-   **Padrões Temporais:** A análise de fraudes ao longo do tempo (`step`) não revelou uma dependência temporal clara, levando à remoção desta coluna do modelo.
-   **Correlação:** Um heatmap de correlação mostrou relações importantes entre o valor da transação e as mudanças de saldo nas contas.

## 🛠️ Metodologia

Para construir o modelo, foi utilizado um `Pipeline` do Scikit-learn, garantindo que o pré-processamento seja aplicado de forma consistente nos dados de treino e teste.

1.  **Seleção de Features:** Foram descartadas as colunas de identificação (`nameOrig`, `nameDest`) e a coluna `isFlaggedFraud`. A coluna `step` também foi removida.
2.  **Divisão dos Dados:** O dataset foi dividido em 70% para treino e 30% para teste, utilizando o parâmetro `stratify=y` para manter a proporção de classes em ambas as amostras.
3.  **Pré-processamento:**
    -   **Features Numéricas** (`amount`, saldos): Padronizadas com `StandardScaler`.
    -   **Features Categóricas** (`type`): Convertidas em variáveis numéricas com `OneHotEncoder`.
4.  **Modelo de Classificação:**
    -   Foi escolhida a **Regressão Logística** pela sua interpretabilidade e eficiência.
    -   Para lidar com o desbalanceamento dos dados, foi utilizado o parâmetro `class_weight="balanced"`, que ajusta os pesos das classes de forma inversamente proporcional às suas frequências.
5.  **Treinamento:** O pipeline completo foi treinado com os dados de treino (`X_train`, `y_train`).

## ⚙️ Requisitos

As bibliotecas necessárias para executar este projeto estão listadas abaixo:

-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `joblib`


Você pode instalá-las com pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## ▶️ Como Executar
1. Dataset: Certifique-se de que o arquivo AIML Dataset.csv está no mesmo diretório do script.
2. Execução: Execute o script Python. Ele realizará todas as etapas de análise, treinamento e avaliação, e salvará o modelo final.
```bash
python fraud_detection.py
```

## 📈 Resultados
O desempenho do modelo é avaliado no conjunto de teste. O script imprime:
   - classification_report: Fornece métricas detalhadas como precisão, recall e f1-score para as classes 0 (não-fraude) e 1 (fraude).
   - confusion_matrix: Mostra o número de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.
Ao final da execução, o pipeline treinado é salvo no arquivo fraud_detection_pipeline.pkl usando joblib, permitindo que seja carregado e utilizado posteriormente para fazer previsões em novos dados sem a necessidade de retreinamento.

---

## Fraud_detection2.py

## Processo de Melhoria do Modelo de Detecção de Fraude (Passos)
1. 📜 Diagnóstico do Problema Inicial
 - Ação: O modelo inicial usava Regressão Logística em uma base de dados onde as fraudes representavam menos de 0.2% do total.
 - Problema: A performance para detectar fraudes era muito baixa. O modelo, ao ser treinado com dados tão desbalanceados, aprendia a simplesmente "ignorar" a classe minoritária, resultando em uma péssima capacidade de detecção.

2. ⚖ Balanceamento dos Dados com SMOTE
 - Ação: Implementamos a técnica SMOTE (Synthetic Minority Over-sampling Technique) para balancear artificialmente o conjunto de treino, mantendo o modelo de Regressão Logística.
 - Resultado:
     - O recall de fraude subiu para 96%, indicando que o modelo agora conseguia identificar quase todas as fraudes reais.
     - A precisão, no entanto, ficou em apenas 2%, o que significava que o modelo gerava uma quantidade enorme de alarmes falsos.
     - Conclusão Parcial: Resolvemos o problema de não encontrar fraudes, mas criamos um problema de excesso de falsos positivos.

3. 🌳 Troca do Algoritmo por Random Forest
- Ação: Mantivemos a estratégia de usar SMOTE, mas trocamos o algoritmo de Regressão Logística por um mais robusto e não-linear, o RandomForestClassifier.
- Resultado Final:
     - O recall de fraude permaneceu excelente em 96%.
     - A precisão de fraude teve um salto impressionante para 64%.
     - O F1-Score (equilíbrio entre precisão e recall) atingiu um ótimo valor de 0.77, demonstrando um modelo eficaz e confiável.

4. Conclusão: Saímos de um modelo inicial ineficaz para um modelo de alta performance, primeiro corrigindo o desbalanceamento dos dados com SMOTE e, em seguida, aplicando um algoritmo mais adequado (Random Forest) para capturar a complexidade do problema.
