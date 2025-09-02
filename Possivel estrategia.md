# Possível estratégia

## Parte 1 — Metodologia e base teórica (para apresentar/explicar)

- Objetivo: evidenciar que adicionar Entropia (H) e Complexidade Estatística (C) às features melhora a separabilidade entre modos de transporte (TMD).
- Dados: `dataset_balanced_w_10_dx_{3,4,5}.csv`; rótulo em `target`.
- Conjuntos de features:
  - Baseline: todas as numéricas, exceto colunas que terminam com `#entropy`/`#complexity`.
  - Baseline+IT: baseline + todas as `#entropy`/`#complexity` (por sensor).
- Higienização: substituição de ±inf por NaN; imputação por mediana; padronização (z-score) para projeção/treino.

### PCA (Principal Component Analysis)

- Ideia: projetar os dados em componentes ortogonais de maior variância, reduzindo dimensionalidade para visualização (2D) sem supervisionar pelos rótulos.
- Passos: (1) padronização; (2) decomposição da matriz de covariância (autovetores/autovalores) ou SVD; (3) projeção nas k primeiras componentes.
- Interpretação: em 2D, clusters menos sobrepostos sugerem melhor separação. Porém, é uma projeção: a separação no espaço original pode ser melhor do que a vista em 2D.

### Métricas

- Acurácia média (Stratified 5-fold) de Regressão Logística — proxy de separabilidade linear no espaço original. Se aumenta com IT, fronteiras ficam mais lineares.
- Silhouette score em PCA 2D — medida ilustrativa de coesão/separação (pode ser negativo em 2D quando há sobreposição após projeção).

### Plano H×C por sensor

- Usamos `sensor#entropy` vs `sensor#complexity` e colorimos por classe para dx=3,4,5. Esse plano mostra como diferentes modos ocupam regiões distintas do espaço de informação.

### Evidência

- Quando comparamos baseline vs baseline+IT:
  - A acurácia média da Regressão Logística aumenta (consistente com melhor separação).
  - Na visualização PCA, há tendência de menos sobreposição entre classes.
  - No plano H×C, observa-se organização por classe, explicando por que IT ajuda.

## Fala sugerida (para apresentação)

“Nosso objetivo foi demonstrar que features de Teoria da Informação — especificamente Entropia de Shannon e Complexidade Estatística — realmente adicionam poder discriminativo ao problema de detecção de modo de transporte. Para isso, separamos as features em dois grupos: um baseline, contendo apenas estatísticas tradicionais, e outro baseline+IT, no qual incorporamos as componentes de entropia e complexidade calculadas por sensor.

Para evidenciar a melhoria, usamos duas perspectivas complementares. Primeiro, uma projeção não supervisionada com PCA em 2 dimensões, que nos permite visualizar se as classes tendem a se organizar melhor quando adicionamos IT. Embora a projeção em 2D não capture toda a separabilidade do espaço original, é possível observar redução de sobreposição entre clusters em vários cenários com IT.

Segundo, adotamos uma métrica objetiva de separabilidade linear no espaço original: a acurácia média de uma Regressão Logística, avaliada com validação cruzada estratificada. De forma consistente, essa acurácia aumenta quando incluímos entropia e complexidade, indicando que as fronteiras de decisão ficam mais lineares.

Também mostramos o plano Entropia × Complexidade por sensor. Esse gráfico revela que diferentes modos ocupam regiões distintas nesse espaço de informação. Essa estrutura é exatamente o que o classificador aproveita quando incluímos as features de IT no vetor de entrada.

Por fim, automatizamos a escolha do melhor sensor para visualização, priorizando sensores relevantes para TMD, como acelerômetro, giroscópio e vetores de rotação. Com isso, os resultados ficam mais alinhados à intuição do domínio e a melhoria trazida pelas features de informação torna-se clara tanto visualmente quanto quantitativamente.”
