# 💻 Classificação de Imagens: Gatos vs. Cachorros com CNN e Transfer Learning

Este projeto implementa uma Rede Neural Convolucional (CNN) para classificar imagens de gatos e cachorros. A abordagem utiliza a técnica de *Transfer Learning* com o modelo pré-treinado **MobileNetV2**, e *Data Augmentation* para melhorar a generalização do modelo.

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Como Funciona](#%EF%B8%8F-como-funciona)
- [Funções Principais](#-funções-principais)
- [Uso para Predição](#-uso-para-predição)

---

## 🚀 Visão Geral

O objetivo deste script é treinar um modelo de aprendizado de máquina capaz de diferenciar imagens de gatos e cachorros com alta acurácia. Para isso, o projeto executa os seguintes passos:

1.  **Download e Preparação:** Baixa e descompacta um dataset público contendo imagens de gatos e cachorros.
2.  **Pré-processamento:** Carrega as imagens em lotes (`batch_size`) e as redimensiona para um tamanho padrão.
3.  **Data Augmentation:** Aplica transformações aleatórias (rotação, zoom, inversão horizontal) nas imagens de treino para aumentar a variabilidade do dataset e evitar overfitting.
4.  **Transfer Learning:** Utiliza a arquitetura do **MobileNetV2** (pré-treinado no dataset ImageNet) como base para extração de características. Seus pesos são congelados para que não sejam alterados durante o treinamento inicial.
5.  **Construção do Modelo:** Adiciona novas camadas no topo da base MobileNetV2:
    -   Uma camada de `GlobalAveragePooling2D` para reduzir a dimensionalidade.
    -   Uma camada de `Dropout` para regularização.
    -   Uma camada `Dense` final com ativação `sigmoid` para a classificação binária.
6.  **Treinamento:** Compila e treina o modelo, utilizando um callback `EarlyStopping` para interromper o treinamento se a perda (`loss`) não melhorar.
7.  **Avaliação:** Mede a acurácia e a perda do modelo em um conjunto de teste separado.
8.  **Predição:** Oferece uma função para classificar novas imagens.

## 🛠️ Como Funciona

O fluxo de trabalho do script é totalmente automatizado, desde a obtenção dos dados até a avaliação final e predição.

1.  **Configuração Inicial:** Define parâmetros globais como dimensões da imagem, tamanho do lote (`batch_size`) e taxa de aprendizado.
2.  **Carregamento de Dados:** Usa a função `image_dataset_from_directory` do TensorFlow para carregar os dados de treino e validação. O dataset de validação é subdividido para criar um conjunto de teste.
3.  **Otimização de Pipeline:** Utiliza `prefetch` com `tf.data.AUTOTUNE` para otimizar o carregamento de dados durante o treinamento.
4.  **Modelo Sequencial:**
    -   A primeira camada (`Rescaling`) normaliza os pixels das imagens para o intervalo `[-1, 1]`.
    -   A camada de `data_augmentation` é aplicada em seguida.
    -   A base `MobileNetV2` (com `trainable = False`) atua como um extrator de características.
    -   As camadas finais realizam a classificação.
5.  **Compilação e Treinamento:** O modelo é compilado com o otimizador `Adam` e a função de perda `BinaryCrossentropy`. O treinamento é executado por um número definido de épocas (`epochs`).
6.  **Visualização de Resultados:** Após o treinamento, gráficos de acurácia e perda (treino vs. validação) são plotados para análise do desempenho.

## 🧠 Funções Principais

- plot_dataset(dataset): Exibe um grid 3x3 de imagens e seus rótulos de um determinado dataset.
- plot_dataset_data_augmentation(dataset): Mostra 9 variações de uma única imagem após aplicar o Data Augmentation.
- plot_model(): Plota os gráficos da acurácia e da perda durante o treinamento e a validação.
- plot_dataset_predictions(dataset): Exibe um grid de imagens do conjunto de teste com as predições feitas pelo modelo.
- predict(image_file): Carrega uma imagem local, faz a predição e imprime o resultado.

## 🎯 Uso para Predição
Após o treinamento, você pode usar a função predict() para classificar suas próprias imagens. Certifique-se de que as imagens estejam no mesmo diretório do script ou forneça o caminho completo.

Exemplo de uso no código:
```python
# Coloque suas imagens no diretório
predict('gatito.jpeg')
predict('cachorrito.jpg')
```
Saída esperada (saídas ilustrativas):

Prediction: 0.00123... | cat

Prediction: 0.98765... | dog
