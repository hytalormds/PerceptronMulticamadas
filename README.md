# Multi-Layer Perceptron (MLP) for MNIST Digit Classification

Este repositório contém um exemplo de implementação de um Perceptron de Múltiplas Camadas (MLP) com duas camadas ocultas para a classificação de dígitos no conjunto de dados MNIST usando TensorFlow. O código está bem comentado para facilitar a compreensão e permitir experimentação com a arquitetura do modelo.

## Pré-requisitos

Antes de executar o código, certifique-se de ter as seguintes bibliotecas instaladas:

```bash
pip install tensorflow tqdm
```

## Como Usar

1. Baixe ou clone este repositório em seu ambiente local.
2. Abra um terminal e navegue até o diretório do projeto.
3. Execute o script Python `PerceptronDeMulticamadas.py`.

```bash
python PerceptronDeMulticamadas.py
```

O modelo será treinado no conjunto de dados MNIST, e a precisão do teste será exibida ao final do treinamento.

## Detalhes do Código

O código é dividido em seções para facilitar a compreensão:

- **Importação de Bibliotecas:** Importa as bibliotecas necessárias, incluindo TensorFlow e tqdm.

- **Carregamento dos Dados:** Utiliza o módulo `input_data` do TensorFlow para carregar o conjunto de dados MNIST.

- **Definição do Modelo:** Cria o grafo computacional para o MLP com duas camadas ocultas.

- **Definição da Arquitetura:** Define as dimensões das camadas e inicializa os pesos e vieses.

- **Treinamento do Modelo:** Usa o otimizador de descida de gradiente para ajustar os pesos com base na entropia cruzada.

- **Teste do Modelo:** Avalia a precisão do modelo no conjunto de teste.

- **Fechamento da Sessão:** Encerra a sessão do TensorFlow após o treinamento.

## Nota

- A precisão de teste pode variar dependendo das configurações específicas de treinamento.
- Este código foi ensinado em um laboratório fornecido pela Duke University pela plataforma Coursera
