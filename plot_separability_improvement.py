import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def select_feature_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (X_baseline, X_with_it, y) from a dataframe.

    - Baseline: todas as colunas numéricas exceto as que terminam com '#entropy' ou '#complexity'.
    - With IT: baseline + colunas que terminam com '#entropy' ou '#complexity'.
    - Remove colunas não preditoras como 'time', 'target', 'user'.
    - Preenche NaNs com a mediana da coluna.
    """
    df = df.copy()

    # Alvo
    if 'target' not in df.columns:
        raise ValueError("Coluna 'target' não encontrada no dataset.")
    y = df['target'].astype(str)

    # Colunas a descartar explicitamente
    drop_cols = {'time', 'target', 'user'}

    # Manter apenas colunas numéricas para features
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Identificar colunas de entropia/complexidade
    it_cols = [c for c in numeric_cols if c.endswith('#entropy') or c.endswith('#complexity')]
    baseline_cols = [c for c in numeric_cols if c not in it_cols]

    # DataFrames de features
    X_baseline = df[baseline_cols].copy()
    X_with_it = df[baseline_cols + it_cols].copy()

    # Tratar NaNs e infinitos com mediana por coluna
    for X in (X_baseline, X_with_it):
        # Substituir +/-inf por NaN, depois preencher com mediana
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Se todas forem NaN numa coluna, preencher com 0 para evitar erro
        medians = X.median(numeric_only=True)
        X.fillna(medians, inplace=True)
        X.fillna(0.0, inplace=True)

    return X_baseline, X_with_it, y


def project_pca(X: pd.DataFrame, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(Xs)
    return Z


def evaluate_linear_separability(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> float:
    """Acurácia média (k-fold) de uma Regressão Logística simples como proxy de separabilidade linear."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    clf = LogisticRegression(max_iter=200, n_jobs=None, random_state=random_state, multi_class='auto')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, Xs, y, cv=cv, scoring='accuracy', n_jobs=None)
    return float(np.mean(scores))


def plot_side_by_side(
    Z_base: np.ndarray,
    Z_it: np.ndarray,
    y: pd.Series,
    sil_base: float,
    sil_it: float,
    acc_base: float,
    acc_it: float,
    title_suffix: str,
    out_path: str,
):
    sns.set(style='whitegrid', context='notebook')
    plt.figure(figsize=(18, 7))

    # Baseline
    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(x=Z_base[:, 0], y=Z_base[:, 1], hue=y, palette='colorblind', s=30, alpha=0.7, linewidth=0, ax=ax1)
    ax1.set_title(f"Baseline (sem IT) | Silhouette={sil_base:.3f} | LR-ACC={acc_base*100:.1f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel('PCA 1')
    ax1.set_ylabel('PCA 2')
    ax1.legend(loc='best', fontsize=8, title='target')

    # With IT
    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(x=Z_it[:, 0], y=Z_it[:, 1], hue=y, palette='colorblind', s=30, alpha=0.7, linewidth=0, ax=ax2)
    ax2.set_title(f"Baseline + IT | Silhouette={sil_it:.3f} | LR-ACC={acc_it*100:.1f}%", fontsize=12, fontweight='bold')
    ax2.set_xlabel('PCA 1')
    ax2.set_ylabel('PCA 2')
    ax2.legend(loc='best', fontsize=8, title='target')

    plt.suptitle(f"Separabilidade por PCA (dx={title_suffix})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs('plan_images', exist_ok=True)

    files = {
        '3': 'dataset_balanced_w_10_dx_3.csv',
        '4': 'dataset_balanced_w_10_dx_4.csv',
        '5': 'dataset_balanced_w_10_dx_5.csv',
    }

    per_dx_figs: List[str] = []
    summary_rows: List[Dict[str, float]] = []

    for dx, path in files.items():
        if not os.path.exists(path):
            print(f"Arquivo não encontrado: {path}")
            continue

        print(f"Processando {path} (dx={dx})...")
        df = pd.read_csv(path)

        X_base, X_it, y = select_feature_sets(df)

        # Evitar erro quando houver 1 classe
        if len(np.unique(y)) < 2:
            print(f"Atenção: menos de 2 classes em dx={dx}; pulando métricas e plot.")
            continue

        # Projeções PCA
        Z_base = project_pca(X_base)
        Z_it = project_pca(X_it)

        # Silhouette (em PCA)
        sil_base = silhouette_score(Z_base, y)
        sil_it = silhouette_score(Z_it, y)

        # Regressão Logística (em feature space original, com padronização interna)
        acc_base = evaluate_linear_separability(X_base, y)
        acc_it = evaluate_linear_separability(X_it, y)

        # Plot por dx
        out_path = os.path.join('plan_images', f'separability_pca_dx{dx}.png')
        plot_side_by_side(Z_base, Z_it, y, sil_base, sil_it, acc_base, acc_it, dx, out_path)
        per_dx_figs.append(out_path)

        summary_rows.append({
            'dx': dx,
            'silhouette_baseline': sil_base,
            'silhouette_with_it': sil_it,
            'lr_acc_baseline': acc_base,
            'lr_acc_with_it': acc_it,
        })

        print(
            f"dx={dx}: Silhouette (base -> IT): {sil_base:.3f} -> {sil_it:.3f} | "
            f"LR-ACC (base -> IT): {acc_base*100:.2f}% -> {acc_it*100:.2f}%"
        )

    # Tabela resumo em CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values('dx')
        summary_csv = os.path.join('plan_images', 'separability_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"Resumo salvo em: {summary_csv}")

    # Grid combinado (3x2) se todos existirem
    if len(per_dx_figs) == 3:
        # Regerar em um só canvas para consistência
        print("Gerando grid combinado...")
        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        for row, dx in enumerate(['3', '4', '5']):
            path = files[dx]
            df = pd.read_csv(path)
            X_base, X_it, y = select_feature_sets(df)
            if len(np.unique(y)) < 2:
                continue
            Z_base = project_pca(X_base)
            Z_it = project_pca(X_it)
            sil_base = silhouette_score(Z_base, y)
            sil_it = silhouette_score(Z_it, y)
            acc_base = evaluate_linear_separability(X_base, y)
            acc_it = evaluate_linear_separability(X_it, y)

            # Baseline
            ax1 = axes[row, 0]
            sns.scatterplot(x=Z_base[:, 0], y=Z_base[:, 1], hue=y, palette='colorblind', s=12, alpha=0.7, linewidth=0, ax=ax1, legend=False)
            ax1.set_title(f"dx={dx} | Baseline | Sil={sil_base:.3f} | ACC={acc_base*100:.1f}%", fontsize=11, fontweight='bold')
            ax1.set_xlabel('PCA 1')
            ax1.set_ylabel('PCA 2')

            # With IT
            ax2 = axes[row, 1]
            sns.scatterplot(x=Z_it[:, 0], y=Z_it[:, 1], hue=y, palette='colorblind', s=12, alpha=0.7, linewidth=0, ax=ax2, legend=False)
            ax2.set_title(f"dx={dx} | Baseline+IT | Sil={sil_it:.3f} | ACC={acc_it*100:.1f}%", fontsize=11, fontweight='bold')
            ax2.set_xlabel('PCA 1')
            ax2.set_ylabel('PCA 2')

        plt.suptitle('Separabilidade por PCA: Baseline vs Baseline+IT (dx=3,4,5)', fontsize=16, fontweight='bold', y=0.92)
        plt.tight_layout()
        combo_path = os.path.join('plan_images', 'separability_pca_dx3_4_5_grid.png')
        plt.savefig(combo_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Grid combinado salvo em: {combo_path}")


if __name__ == '__main__':
    main()


