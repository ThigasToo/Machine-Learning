# 📈 LSTM para Previsão de Preços de Criptomoeda

## 📌 Descrição
Este projeto implementa um modelo **LSTM (Long Short-Term Memory)** para prever os preços de fechamento de uma criptomoeda (Bitcoin) com base em dados históricos.  
O script **`lstm_bit2.py`** carrega os dados de preços, realiza pré-processamento, treina o modelo e gera previsões tanto para o conjunto de teste quanto para datas futuras.

---

## 🛠️ Estrutura do Código

1. **Carregamento e Visualização dos Dados**
   - Leitura do arquivo `bit4.csv`.
   - Análise exploratória (`head`, `info`, `describe`).
   - Gráficos de preços de abertura e fechamento ao longo do tempo.

2. **Pré-processamento**
   - Seleção da coluna de preços de fechamento (`Último`).
   - Escalonamento com `StandardScaler`.
   - Criação de janelas de 60 dias para treino (sliding window).

3. **Modelo LSTM**
   - Arquitetura baseada em camadas LSTM + Dense.
   - Função de perda: `mean_squared_error`.
   - Métrica: `RootMeanSquaredError`.
   - Treinamento com 50 épocas e batch size de 32.

4. **Avaliação**
   - Previsões sobre o conjunto de teste.
   - Gráficos comparando **dados reais vs previsões**.

5. **Previsão Futura**
   - Projeção de **30 dias à frente** com base nos últimos 60 valores conhecidos.
   - Geração de um DataFrame com datas futuras e preços previstos.
   - Plotagem dos resultados com curva histórica + previsões futuras.

---

## 📊 Exemplos de Saída

- **Modelo e Treinamento**
```python
model = keras.models.Sequential()
#First Layer
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#Second Layer
model.add(keras.layers.LSTM(units=64, return_sequences=False))
#3rd Layer
model.add(keras.layers.Dense(units=128, activation='relu'))
#4th Layer (dropout)
model.add(keras.layers.Dense(units=128, activation='relu'))
#Final Layer
model.add(keras.layers.Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError])

training = model.fit(X_train, y_train, batch_size=32, epochs=50)
```

- **Previsões Futuras**
```python
plt.figure(figsize=(14, 8))
plt.plot(data['Data'], data['Último'], label='Fechamento Real', color='blue')
plt.plot(test['Data'], test['Predictions'], label='Previsões (teste)', color='red')
plt.plot(future_df['Data'].astype(str), future_df['Previsão_Futura'], label='Previsão Futura', color='orange', linestyle='--')
plt.title('Preço de Fechamento com Previsões Futuras')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.show()
```

## 🚀 Como Usar
1. Garanta que o arquivo bit4.csv está no mesmo diretório do script.

2. Instale as dependências necessárias:
  ```bash
 pip install tensorflow pandas seaborn matplotlib numpy scikit-learn
```
3. Execute o script:
 ```bash
 python lstm_bit2.py
```
## 📈 Insights Possíveis
- Identificação da tendência de preços ao longo do tempo.
- Comparação entre previsões e valores reais no período de teste.
- Projeções de preços futuros para apoiar análises de investimento.
plt.legend()
plt.show()
