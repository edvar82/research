import os
import argparse
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


def select_sensor_feature_sets(df: pd.DataFrame, sensor: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Seleciona features apenas do sensor informado.

    - Baseline: colunas do sensor (prefixo f"{sensor}#") exceto '#entropy' e '#complexity'.
    - With IT: baseline + as colunas do sensor que terminam com '#entropy' e '#complexity'.
    - Alvo: 'target'.
    - Preenche NaNs/inf com mediana por coluna.
    """
    df = df.copy()
    if 'target' not in df.columns:
        raise ValueError("Coluna 'target' não encontrada no dataset.")
    y = df['target'].astype(str)

    prefix = f"{sensor}#"
    sensor_cols = [c for c in df.columns if c.startswith(prefix)]

    it_cols = [c for c in sensor_cols if c.endswith('#entropy') or c.endswith('#complexity')]
    baseline_cols = [c for c in sensor_cols if c not in it_cols]

    baseline_cols = [c for c in baseline_cols if pd.api.types.is_numeric_dtype(df[c])]
    it_cols = [c for c in it_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not baseline_cols:
        raise ValueError(f"Nenhuma coluna baseline encontrada para o sensor: {sensor}")

    X_baseline = df[baseline_cols].copy()
    X_with_it = df[baseline_cols + it_cols].copy() if it_cols else X_baseline.copy()

    for X in (X_baseline, X_with_it):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        medians = X.median(numeric_only=True)
        X.fillna(medians, inplace=True)
        X.fillna(0.0, inplace=True)

    return X_baseline, X_with_it, y


def project_pca(X: pd.DataFrame, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    n_samples, n_features = X.shape
    max_comps = max(1, min(n_components, n_features, max(1, n_samples - 1)))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=max_comps, random_state=random_state)
    return pca.fit_transform(Xs)


def evaluate_lr_acc(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> float:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    clf = LogisticRegression(max_iter=200, random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, Xs, y, cv=cv, scoring='accuracy')
    return float(np.mean(scores))


def plot_pca_side_by_side(
    Z_base: np.ndarray,
    Z_it: np.ndarray,
    y: pd.Series,
    sil_base: float,
    sil_it: float,
    acc_base: float,
    acc_it: float,
    sensor: str,
    dx: str,
    out_path: str,
):
    sns.set(style='whitegrid', context='notebook')
    plt.figure(figsize=(18, 7))

    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(x=Z_base[:, 0], y=Z_base[:, 1], hue=y, palette='colorblind', s=30, alpha=0.7, linewidth=0, ax=ax1)
    ax1.set_title(f"{sensor} | Baseline (dx={dx}) | Sil={sil_base:.3f} | ACC={acc_base*100:.1f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel('PCA 1')
    ax1.set_ylabel('PCA 2')
    ax1.legend(loc='best', fontsize=8, title='target')

    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(x=Z_it[:, 0], y=Z_it[:, 1], hue=y, palette='colorblind', s=30, alpha=0.7, linewidth=0, ax=ax2)
    ax2.set_title(f"{sensor} | Baseline+IT (dx={dx}) | Sil={sil_it:.3f} | ACC={acc_it*100:.1f}%", fontsize=12, fontweight='bold')
    ax2.set_xlabel('PCA 1')
    ax2.set_ylabel('PCA 2')
    ax2.legend(loc='best', fontsize=8, title='target')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_hc_grid(datasets: Dict[str, pd.DataFrame], sensor: str, out_path: str) -> None:
    entropy_col = f"{sensor}#entropy"
    complexity_col = f"{sensor}#complexity"

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    for i, (dx, df) in enumerate(sorted(datasets.items(), key=lambda x: int(x[0]))):
        ax = axes[i]
        if entropy_col not in df.columns or complexity_col not in df.columns:
            ax.set_visible(False)
            continue
        sns.scatterplot(
            data=df,
            x=entropy_col,
            y=complexity_col,
            hue='target',
            palette='colorblind',
            s=30,
            alpha=0.7,
            linewidth=0,
            ax=ax,
        )
        ax.set_title(f"dx={dx}", fontweight='bold')
        ax.set_xlabel('Shannon Entropy')
        if i == 0:
            ax.set_ylabel('Statistical Complexity')
        else:
            ax.set_ylabel('')
        ax.legend_.remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title='target', loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"{sensor} | Plano Entropia × Complexidade (dx=3,4,5)", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def safe_name(sensor: str) -> str:
    return sensor.replace('.', '_').replace('/', '_')


def list_users(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    users: set[str] = set()
    for df in dfs.values():
        if 'user' in df.columns:
            vals = df['user'].dropna().astype(str).unique().tolist()
            users.update(vals)
    return sorted(users)


def list_candidate_sensors(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    sensors: set[str] = set()
    for df in dfs.values():
        for c in df.columns:
            if '#' in c:
                prefix = c.split('#', 1)[0]
                if prefix not in {'time', 'target', 'user'}:
                    sensors.add(prefix)
    return sorted(sensors)


def score_sensor_across_dx(dfs: Dict[str, pd.DataFrame], sensor: str, min_rows: int = 50, random_state: int = 42) -> Tuple[float, float, int]:
    delta_acc: List[float] = []
    delta_sil: List[float] = []
    for dx, df in dfs.items():
        if len(df) < min_rows:
            continue
        entropy_col = f"{sensor}#entropy"
        complexity_col = f"{sensor}#complexity"
        if (entropy_col not in df.columns) and (complexity_col not in df.columns):
            continue
        try:
            X_base, X_it, y = select_sensor_feature_sets(df, sensor)
        except ValueError:
            continue
        if len(np.unique(y)) < 2:
            continue
        if X_base.shape[1] < 2:
            continue
        Z_base = project_pca(X_base, random_state=random_state)
        Z_it = project_pca(X_it, random_state=random_state)
        sil_b = silhouette_score(Z_base, y)
        sil_i = silhouette_score(Z_it, y)
        acc_b = evaluate_lr_acc(X_base, y, random_state=random_state)
        acc_i = evaluate_lr_acc(X_it, y, random_state=random_state)
        delta_acc.append(acc_i - acc_b)
        delta_sil.append(sil_i - sil_b)
    if not delta_acc:
        return 0.0, 0.0, 0
    return float(np.mean(delta_acc)), float(np.mean(delta_sil)), len(delta_acc)


def main():
    parser = argparse.ArgumentParser(description='Gera plots para os melhores sensores por usuário (TMD).')
    parser.add_argument('--per-user-top-k', type=int, default=6, help='Quantidade de melhores sensores por usuário.')
    parser.add_argument('--outdir', type=str, default='new_plots', help='Diretório base de saída.')
    parser.add_argument('--min-rows', type=int, default=50, help='Mínimo de linhas por dx por usuário.')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    files = {
        '3': 'dataset_balanced_w_10_dx_3.csv',
        '4': 'dataset_balanced_w_10_dx_4.csv',
        '5': 'dataset_balanced_w_10_dx_5.csv',
    }

    # Carregar datasets completos (sem filtro de usuário ainda)
    all_dfs: Dict[str, pd.DataFrame] = {}
    for dx, path in files.items():
        if not os.path.exists(path):
            print(f"Arquivo não encontrado: {path}")
            continue
        all_dfs[dx] = pd.read_csv(path)

    if not all_dfs:
        raise SystemExit('Nenhum dataset encontrado.')

    # Usuários no dataset
    users = list_users(all_dfs)
    if not users:
        raise SystemExit('Nenhum usuário encontrado na coluna user.')

    # Sensores candidatos relevantes para TMD
    relevance_allow = {
        'android.sensor.accelerometer',
        'android.sensor.linear_acceleration',
        'android.sensor.gyroscope',
        'android.sensor.gyroscope_uncalibrated',
        'android.sensor.rotation_vector',
        'android.sensor.game_rotation_vector',
        'android.sensor.gravity',
        'android.sensor.magnetic_field',
        'android.sensor.magnetic_field_uncalibrated',
        'android.sensor.orientation',
        'android.sensor.light',
    }

    base_candidates = [c for c in list_candidate_sensors(all_dfs) if c in relevance_allow]
    if not base_candidates:
        raise SystemExit('Nenhum sensor relevante encontrado.')

    # Processar por usuário
    for user in users:
        user_out = os.path.join(outdir, str(user))
        os.makedirs(user_out, exist_ok=True)
        # Subpastas para separar tipos de saída
        hc_dir = os.path.join(user_out, 'hc')
        pca_dir = os.path.join(user_out, 'pca')
        csv_dir = os.path.join(user_out, 'csv')
        os.makedirs(hc_dir, exist_ok=True)
        os.makedirs(pca_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        print(f"Processando usuário: {user}")

        # Filtrar dados por usuário
        user_dfs: Dict[str, pd.DataFrame] = {}
        for dx, df in all_dfs.items():
            if 'user' in df.columns:
                df_u = df[df['user'] == user].copy()
            else:
                df_u = df.copy()
            user_dfs[dx] = df_u

        # Rankear sensores para este usuário
        ranking: List[Tuple[str, float, float, int]] = []
        for s in base_candidates:
            d_acc, d_sil, n = score_sensor_across_dx(user_dfs, s, min_rows=args.min_rows, random_state=args.random_state)
            ranking.append((s, d_acc, d_sil, n))
        ranking.sort(key=lambda t: (t[1], t[2]), reverse=True)

        # Salvar ranking
        import csv
        rank_path = os.path.join(csv_dir, 'sensor_improvement_ranking.csv')
        with open(rank_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sensor', 'mean_delta_lr_acc', 'mean_delta_silhouette', 'num_dx'])
            for s, da, ds, n in ranking:
                writer.writerow([s, f"{da:.4f}", f"{ds:.4f}", n])
        print(f"Ranking salvo em: {rank_path}")

        if not ranking or ranking[0][3] == 0:
            print(f"Usuário {user}: nenhum sensor com dados suficientes.")
            continue

        top_k = max(1, args.per_user_top_k)
        top_sensors = [r[0] for r in ranking[:top_k]]
        print(f"Usuário {user}: top {top_k} sensores: {', '.join(top_sensors)}")

        # Gerar saídas para os top sensores deste usuário
        for sensor in top_sensors:
            summary_rows: List[Dict[str, float]] = []
            for dx, df_u in user_dfs.items():
                if len(df_u) < args.min_rows:
                    print(f"{user} | {sensor} | dx={dx}: menos de {args.min_rows} linhas; pulando.")
                    continue
                try:
                    X_base, X_it, y = select_sensor_feature_sets(df_u, sensor)
                except ValueError as e:
                    print(f"{user} | {sensor} | dx={dx}: {e}")
                    continue
                if len(np.unique(y)) < 2:
                    print(f"{user} | {sensor} | dx={dx}: menos de 2 classes; pulando.")
                    continue
                Z_base = project_pca(X_base, random_state=args.random_state)
                Z_it = project_pca(X_it, random_state=args.random_state)
                sil_base = silhouette_score(Z_base, y)
                sil_it = silhouette_score(Z_it, y)
                acc_base = evaluate_lr_acc(X_base, y, random_state=args.random_state)
                acc_it = evaluate_lr_acc(X_it, y, random_state=args.random_state)
                out_path = os.path.join(pca_dir, f"{safe_name(sensor)}_separability_pca_dx{dx}.png")
                plot_pca_side_by_side(Z_base, Z_it, y, sil_base, sil_it, acc_base, acc_it, sensor, dx, out_path)
                summary_rows.append({
                    'sensor': sensor,
                    'dx': dx,
                    'silhouette_baseline': sil_base,
                    'silhouette_with_it': sil_it,
                    'lr_acc_baseline': acc_base,
                    'lr_acc_with_it': acc_it,
                })
                print(f"{user} | {sensor} | dx={dx}: Sil (b->it): {sil_base:.3f}->{sil_it:.3f} | ACC (b->it): {acc_base*100:.2f}%->{acc_it*100:.2f}%")

            # CSV resumo por sensor
            if summary_rows:
                df_sum = pd.DataFrame(summary_rows).sort_values('dx')
                csv_path = os.path.join(csv_dir, f"{safe_name(sensor)}_separability_summary.csv")
                df_sum.to_csv(csv_path, index=False)
                print(f"Resumo salvo em: {csv_path}")

            # Plano H×C
            hc_path = os.path.join(hc_dir, f"{safe_name(sensor)}_entropy_complexity_dx3_4_5_grid.png")
            plot_hc_grid(user_dfs, sensor, hc_path)
            print(f"Plano H×C salvo em: {hc_path}")


if __name__ == '__main__':
    main()


