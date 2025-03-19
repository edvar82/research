# Comparação de desempenho do modelo 10SecondWindow_MLP

## Comparativo entre literatura e implementações com entropia e complexidade

<div style="page-break-inside: avoid; margin-bottom: 30px;">

### Literatura vs. Implementações com mesmos parâmetros

| Implementação                      | dx  | Batch Size | Epochs | Accuracy | Loss   | Precision | Recall | F1 Score |
| ---------------------------------- | --- | ---------- | ------ | -------- | ------ | --------- | ------ | -------- |
| **Literatura**                     | -   | 128        | 60     | 0.9220   | 0.2640 | 0.9321    | 0.9084 | 0.9201   |
| **Adding Entropia e Complexidade** | 3   | 128        | 60     | 0.9154   | 0.2845 | 0.9245    | 0.9046 | 0.9145   |
| **Adding Entropia e Complexidade** | 4   | 128        | 60     | 0.9062   | 0.2635 | 0.9167    | 0.8969 | 0.9067   |
| **Adding Entropia e Complexidade** | 5   | 128        | 60     | 0.9169   | 0.2667 | 0.9218    | 0.9062 | 0.9139   |

</div>

<div style="page-break-inside: avoid; margin-bottom: 30px; margin-top: 30px;">

### Implementações otimizadas com entropia e complexidade

| Implementação                      | dx  | Batch Size | Epochs | Accuracy | Loss   | Precision | Recall | F1 Score |
| ---------------------------------- | --- | ---------- | ------ | -------- | ------ | --------- | ------ | -------- |
| **Literatura**                     | -   | 128        | 60     | 0.9220   | 0.2640 | 0.9321    | 0.9084 | 0.9201   |
| **Adding Entropia e Complexidade** | 3   | 64         | 200    | 0.9385   | 0.2255 | 0.9412    | 0.9354 | 0.9383   |
| **Adding Entropia e Complexidade** | 4   | 32         | 60     | 0.9308   | 0.2425 | 0.9350    | 0.9292 | 0.9321   |
| **Adding Entropia e Complexidade** | 5   | 64         | 100    | 0.9400   | 0.2425 | 0.9412    | 0.9354 | 0.9383   |

</div>

<div style="page-break-before: always;"></div>

<div style="page-break-inside: avoid !important; margin-bottom: 30px; margin-top: 30px; break-inside: avoid;">

## Resumo dos melhores resultados

| Configuração    | Parâmetros           | Accuracy | F1 Score |
| --------------- | -------------------- | -------- | -------- |
| **Literatura**  | Batch=128, Epochs=60 | 0.9220   | 0.9201   |
| **Melhor dx=3** | Batch=64, Epochs=200 | 0.9385   | 0.9383   |
| **Melhor dx=4** | Batch=32, Epochs=60  | 0.9308   | 0.9321   |
| **Melhor dx=5** | Batch=64, Epochs=100 | 0.9400   | 0.9383   |

</div>

<div style="margin-bottom: 40px;">

## Conclusões

- Os modelos com entropia e complexidade e parâmetros otimizados consistentemente superam os modelos da literatura em todas as métricas, dada as mudanças na forma de treinamento, alterando o batch size e o número de epochs.
- Com os mesmos parâmetros da literatura (`batch size = 128`, `epochs = 60`), a implementação com `dx = 5` obteve resultados mais próximos da literatura.
- A melhor performance foi obtida com `window_size = 10`, `dx = 5`, usando `batch size = 64` e `epochs = 100`, com acurácia de 94%, precisão de 94.12%, recall de 93.54% e F1 score de 93.83%.
- O dataset com um `window size = 5` sem entropia e complexidade tem ~6000 linhas; com entropia e complexidade e `window size = 10` tem ~3000 linhas, reduzindo o tempo de treinamento e processamento, tendo uma dimininuição de ~50% no tamanho do dataset, implicando em uma diminuição do tempo de treinamento e de processamento.

> **Nota**: `dx` é a janela deslizante de tempo, que é o tamanho da janela de tempo que o modelo usa para prever a próxima amostra.

</div>

<div style="page-break-before: always;"></div>

# Comparação de desempenho do modelo 10SecondWindow_MLP_Federated

<div style="page-break-inside: avoid; margin-bottom: 40px;">

## Resumo comparativo

| Modelo                             | Comm Round | Accuracy | Loss  |
| ---------------------------------- | ---------- | -------- | ----- |
| **Literatura**                     | 199        | 85.751%  | 1.091 |
| **Entropia e Complexidade (dx=3)** | 199        | 89.538%  | 1.061 |
| **Entropia e Complexidade (dx=4)** | 199        | 90.000%  | 1.058 |
| **Entropia e Complexidade (dx=5)** | 199        | 87.846%  | 1.072 |

</div>

## Detalhes das implementações

<div style="page-break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">

### Literatura

- **Accuracy**: 85.751%
- **Loss**: 1.091
- **Métricas por Classe**:

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.76      | 0.89   | 0.82     | 237     |
| 1                | 0.92      | 0.83   | 0.87     | 223     |
| 2                | 0.86      | 0.84   | 0.85     | 247     |
| 3                | 0.84      | 0.84   | 0.84     | 209     |
| 4                | 0.92      | 0.88   | 0.90     | 263     |
| **Macro avg**    | 0.86      | 0.86   | 0.86     | 1179    |
| **Weighted avg** | 0.86      | 0.86   | 0.86     | 1179    |

</div>

<div style="page-break-before: always;"></div>

<div style="page-break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">

### Entropia e Complexidade (dx=3)

- **Accuracy**: 89.538%
- **Loss**: 1.061
- **Métricas por Classe**:

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.83      | 0.92   | 0.87     | 113     |
| 1                | 0.92      | 0.90   | 0.91     | 144     |
| 2                | 0.87      | 0.94   | 0.90     | 112     |
| 3                | 0.92      | 0.85   | 0.88     | 138     |
| 4                | 0.93      | 0.89   | 0.91     | 143     |
| **Macro avg**    | 0.89      | 0.90   | 0.89     | 650     |
| **Weighted avg** | 0.90      | 0.90   | 0.90     | 650     |

</div>

<div style="page-break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">

### Entropia e Complexidade (dx=4)

- **Accuracy**: 90.000%
- **Loss**: 1.058
- **Métricas por Classe**:

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.88      | 0.97   | 0.92     | 139     |
| 1                | 0.89      | 0.85   | 0.87     | 119     |
| 2                | 0.90      | 0.89   | 0.90     | 131     |
| 3                | 0.86      | 0.81   | 0.83     | 120     |
| 4                | 0.96      | 0.96   | 0.96     | 141     |
| **Macro avg**    | 0.90      | 0.90   | 0.90     | 650     |
| **Weighted avg** | 0.90      | 0.90   | 0.90     | 650     |

</div>

<div style="page-break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">

<div style="page-break-inside: avoid !important; break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">

<div style="page-break-before: always;"></div>

### Entropia e Complexidade (dx=5)

- **Accuracy**: 87.846%
- **Loss**: 1.072
- **Métricas por Classe**:

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.81      | 0.91   | 0.85     | 117     |
| 1                | 0.88      | 0.88   | 0.88     | 130     |
| 2                | 0.85      | 0.94   | 0.90     | 123     |
| 3                | 0.92      | 0.75   | 0.82     | 146     |
| 4                | 0.93      | 0.93   | 0.93     | 134     |
| **Macro avg**    | 0.88      | 0.88   | 0.88     | 650     |
| **Weighted avg** | 0.88      | 0.88   | 0.88     | 650     |

</div>

<div style="margin-top: 30px;">

## Conclusões

- Os modelos federados com entropia e complexidade apresentam desempenho superior ao modelo da literatura, com aumento de até 4.25 pontos percentuais na acurácia (85.75% para 90.00%).
- A melhor configuração foi obtida com `dx = 4`, alcançando 90% de acurácia.
- As implementações com entropia e complexidade conseguem manter um bom equilíbrio entre precision e recall.
- O modelo com `dx = 4` apresenta melhor desempenho na classe 4 (F1-Score de 0.96) e na classe 0 (recall de 0.97).
- A redução do tamanho do dataset com a adição de entropia e complexidade (~50% menos dados) não prejudicou o desempenho, pelo contrário, melhorou os resultados.
- O aprendizado federado com estas características mostrou-se eficiente para este tipo de classificação, preservando a privacidade dos dados distribuídos entre os clientes.

</div>

<div style="page-break-before: always;"></div>

# Comparação de desempenho do modelo 10SecondWindow*FederatedEnsemble*(CM)

<div style="page-break-inside: avoid; margin-bottom: 40px;">

## Composição do modelo

O modelo de ensemble federado utiliza uma combinação dos seguintes classificadores:

- SimpleMLP
- XGBClassifier
- RandomForestClassifier

## Resumo comparativo

| Modelo                             | Comm Round | Accuracy |
| ---------------------------------- | ---------- | -------- |
| **Literatura**                     | 199        | 95.081%  |
| **Entropia e Complexidade (dx=3)** | 199        | 95.077%  |
| **Entropia e Complexidade (dx=4)** | 199        | 92.462%  |
| **Entropia e Complexidade (dx=5)** | 199        | 94.923%  |

</div>

## Detalhes das implementações

<div style="page-break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">

### Literatura

- **Accuracy**: 95.081%
- **Métricas por Classe**:

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.92      | 0.95   | 0.93     | 242     |
| 1                | 0.98      | 0.91   | 0.94     | 235     |
| 2                | 0.96      | 0.97   | 0.97     | 236     |
| 3                | 0.96      | 0.94   | 0.95     | 238     |
| 4                | 0.94      | 0.98   | 0.96     | 228     |
| **Macro avg**    | 0.95      | 0.95   | 0.95     | 1179    |
| **Weighted avg** | 0.95      | 0.95   | 0.95     | 1179    |

</div>

<div style="page-break-before: always;"></div>

<div style="page-break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">
<div style="page-break-before: always;"></div>

### Entropia e Complexidade (dx=3)

- **Accuracy**: 95.077%
- **Métricas por Classe**:

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.95      | 0.95   | 0.95     | 127     |
| 1                | 0.95      | 0.90   | 0.93     | 134     |
| 2                | 0.95      | 0.99   | 0.97     | 127     |
| 3                | 0.96      | 0.94   | 0.95     | 140     |
| 4                | 0.94      | 0.97   | 0.96     | 122     |
| **Macro avg**    | 0.95      | 0.95   | 0.95     | 650     |
| **Weighted avg** | 0.95      | 0.95   | 0.95     | 650     |

</div>

<div style="page-break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">

### Entropia e Complexidade (dx=4)

- **Accuracy**: 92.462%
- **Métricas por Classe**:

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.88      | 0.93   | 0.91     | 130     |
| 1                | 0.95      | 0.84   | 0.89     | 125     |
| 2                | 0.92      | 0.97   | 0.94     | 131     |
| 3                | 0.98      | 0.92   | 0.95     | 136     |
| 4                | 0.90      | 0.96   | 0.93     | 128     |
| **Macro avg**    | 0.93      | 0.92   | 0.92     | 650     |
| **Weighted avg** | 0.93      | 0.92   | 0.92     | 650     |

</div>

<div style="page-break-before: always;"></div>

<div style="page-break-inside: avoid !important; break-inside: avoid; margin-top: 30px; margin-bottom: 40px;">

### Entropia e Complexidade (dx=5)

- **Accuracy**: 94.923%
- **Métricas por Classe**:

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.85      | 0.95   | 0.90     | 108     |
| 1                | 0.90      | 0.87   | 0.89     | 130     |
| 2                | 0.93      | 0.91   | 0.92     | 140     |
| 3                | 0.99      | 0.90   | 0.94     | 131     |
| 4                | 0.91      | 0.96   | 0.93     | 141     |
| **Macro avg**    | 0.92      | 0.92   | 0.92     | 650     |
| **Weighted avg** | 0.92      | 0.92   | 0.92     | 650     |

</div>

<div style="margin-top: 30px;">

## Conclusões

- A aplicação de entropia e complexidade com `dx = 3` manteve o desempenho praticamente idêntico à literatura (95.08% vs 95.08%).
- Mesmo com a redução de aproximadamente 50% no tamanho do dataset, o desempenho se manteve em níveis excelentes.
- Modelos com `dx = 4` e `dx = 5` apresentaram pequena redução no desempenho, mas ainda mantiveram acurácia acima de 92%.

</div>