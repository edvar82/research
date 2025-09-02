# Evidência de melhoria com Entropia e Complexidade (IT) na separação de modos de transporte

## Objetivo

Demonstrar visual e quantitativamente como a adição das features de Teoria da Informação (Shannon Entropy e Statistical Complexity) melhora a separabilidade entre os modos de transporte.

## Dados e preparação

- Datasets: `dataset_balanced_w_10_dx_{3,4,5}.csv`.
- Alvo: coluna `target`.
- Baseline (sem IT): todas as colunas numéricas exceto as que terminam com `#entropy` ou `#complexity`.
- Baseline+IT: baseline + todas as colunas que terminam com `#entropy` e `#complexity`.
- Higienização: substituição de ±inf por NaN e imputação por mediana de coluna; padronização z-score (quando aplicável).

## Técnica de análise

- Projeção 2D com PCA para visualização qualitativa da separação (baseline vs baseline+IT), colorindo por `target`.
- Métrica de separabilidade linear no espaço original das features: acurácia média (Stratified 5-fold) de Regressão Logística (proxy direta da separabilidade linear entre as classes).
- Métrica de agrupamento em 2D (ilustrativa): Silhouette score na projeção PCA (pode ser negativo quando as classes se sobrepõem no 2D, mesmo com separabilidade no espaço de maior dimensão).

## Scripts (reprodutibilidade)

- Comparação geral (todas as features):
  - Script: `plot_separability_improvement.py`
  - Saídas:
    - `plan_images/separability_pca_dx3.png`
    - `plan_images/separability_pca_dx4.png`
    - `plan_images/separability_pca_dx5.png`
    - `plan_images/separability_pca_dx3_4_5_grid.png`
    - `plan_images/separability_summary.csv`
- Por sensor específico (ex.: `android.sensor.orientation`):
  - Script: `plot_separability_by_sensor.py`
  - Execução: `python plot_separability_by_sensor.py --sensor android.sensor.orientation`
  - Saídas:
    - `plan_images/android_sensor_orientation_separability_pca_dx3.png`
    - `plan_images/android_sensor_orientation_separability_pca_dx4.png`
    - `plan_images/android_sensor_orientation_separability_pca_dx5.png`
    - `plan_images/android_sensor_orientation_separability_summary.csv`
    - `plan_images/android_sensor_orientation_entropy_complexity_dx3_4_5_grid.png` (plano H×C do sensor)

## Resultados observados

- Geral (multimodal, todas as features):
  - dx=3: LR-ACC baseline→IT ≈ 81.6% → 85.4%
  - dx=4: LR-ACC baseline→IT ≈ 82.5% → 86.4%
  - dx=5: LR-ACC baseline→IT ≈ 81.6% → 84.9%
  - Interpretação: melhora consistente de separabilidade linear quando IT é incluído.
- Sensor `android.sensor.orientation` (apenas features do sensor):
  - dx=3: LR-ACC baseline→IT ≈ 36.8% → 37.8%
  - dx=4: LR-ACC baseline→IT ≈ 36.4% → 38.1%
  - dx=5: LR-ACC baseline→IT ≈ 34.8% → 36.6%
  - Interpretação: ganhos modestos, porém positivos; como esperado, um único sensor possui menor poder discriminativo do que o conjunto multimodal, mas IT agrega informação útil.

## Como as figuras evidenciam a melhora

- Nos painéis PCA (baseline vs baseline+IT), há redução da sobreposição de clusters e maior organização por classe quando IT é adicionado. Os títulos mostram Silhouette e LR-ACC; mesmo quando o Silhouette em 2D é baixo/negativo (devido à projeção), a LR-ACC aumenta com IT, indicando fronteiras mais lineares no espaço original das features.
- O plano Entropia × Complexidade do sensor (`Entropy × Complexity`) mostra que diferentes modos ocupam regiões distintas do espaço de informação. Essa estrutura adicional é incorporada ao vetor de features, melhorando a classificação.

## Conclusão

A inclusão de entropia e complexidade melhora a separabilidade entre modos de transporte. Isso é suportado por:

- Incrementos consistentes de acurácia de um classificador linear (proxy da separabilidade) em todos os `dx` no cenário multimodal.
- Melhor organização visual na projeção PCA quando IT é adicionada e diferenciação de regiões no plano H×C.
