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
    - Remove colunas não preditoras: 'time', 'target', 'user'.
    - Preenche NaNs/inf com mediana por coluna.
    """
    df = df.copy()

    if 'target' not in df.columns:
        raise ValueError("Coluna 'target' não encontrada no dataset.")
    y = df['target'].astype(str)

    prefix = f"{sensor}#"
    sensor_cols = [c for c in df.columns if c.startswith(prefix)]

    # Separar IT e baseline
    it_cols = [c for c in sensor_cols if c.endswith('#entropy') or c.endswith('#complexity')]
    baseline_cols = [c for c in sensor_cols if c not in it_cols]

    # Apenas colunas numéricas
    baseline_cols = [c for c in baseline_cols if pd.api.types.is_numeric_dtype(df[c])]
    it_cols = [c for c in it_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not baseline_cols:
        raise ValueError(f"Nenhuma coluna baseline encontrada para o sensor: {sensor}")

    X_baseline = df[baseline_cols].copy()
    X_with_it = df[baseline_cols + it_cols].copy() if it_cols else X_baseline.copy()

    # Limpeza de NaN/inf
    for X in (X_baseline, X_with_it):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        medians = X.median(numeric_only=True)
        X.fillna(medians, inplace=True)
        X.fillna(0.0, inplace=True)

    return X_baseline, X_with_it, y


def project_pca(X: pd.DataFrame, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Projeta em PCA com número de componentes ajustado à dimensionalidade disponível."""
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
    """Plano Entropia × Complexidade para o sensor em dx=3,4,5."""
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


def _list_candidate_sensors(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    sensors: set[str] = set()
    for df in dfs.values():
        for c in df.columns:
            if '#' in c:
                prefix = c.split('#', 1)[0]
                if prefix not in {'time', 'target', 'user'}:
                    sensors.add(prefix)
    return sorted(sensors)


def _score_sensor_across_dx(dfs: Dict[str, pd.DataFrame], sensor: str, random_state: int = 42, min_rows: int = 200) -> Tuple[float, float, int]:
    """Retorna (delta_acc_medio, delta_sil_medio, num_dx_validos) para um sensor.
    delta = (with_it - baseline)."""
    delta_acc_list: List[float] = []
    delta_sil_list: List[float] = []
    for dx, df in dfs.items():
        # Requisitos mínimos por dx
        if len(df) < min_rows:
            continue
        entropy_col = f"{sensor}#entropy"
        complexity_col = f"{sensor}#complexity"
        if (entropy_col not in df.columns) and (complexity_col not in df.columns):
            # sem IT específico do sensor neste dx
            continue
        try:
            X_base, X_it, y = select_sensor_feature_sets(df, sensor)
        except ValueError:
            continue
        if len(np.unique(y)) < 2:
            continue
        # baseline precisa ter pelo menos 2 features para um PCA significativo
        if X_base.shape[1] < 2:
            continue
        Z_base = project_pca(X_base, random_state=random_state)
        Z_it = project_pca(X_it, random_state=random_state)
        sil_base = silhouette_score(Z_base, y)
        sil_it = silhouette_score(Z_it, y)
        acc_base = evaluate_lr_acc(X_base, y, random_state=random_state)
        acc_it = evaluate_lr_acc(X_it, y, random_state=random_state)
        delta_acc_list.append(acc_it - acc_base)
        delta_sil_list.append(sil_it - sil_base)
    if not delta_acc_list:
        return 0.0, 0.0, 0
    return float(np.mean(delta_acc_list)), float(np.mean(delta_sil_list)), len(delta_acc_list)


def main():
    parser = argparse.ArgumentParser(description='Visualização de separabilidade por sensor (Baseline vs IT).')
    parser.add_argument('--sensor', type=str, required=False, help='Nome do sensor, ex: android.sensor.orientation')
    parser.add_argument('--auto-best', action='store_true', help='Seleciona automaticamente o sensor com maior melhoria (LR-ACC).')
    parser.add_argument('--user', type=str, required=False, help='Filtra por um usuário específico (coluna user), ex: U1')
    parser.add_argument('--min-rows', type=int, default=200, help='Mínimo de linhas por dx para considerar (ajuste menor ao filtrar por usuário).')
    parser.add_argument('--outdir', type=str, default='new_plots', help='Diretório de saída para salvar gráficos e CSVs.')
    parser.add_argument('--top-k', type=int, default=4, help='Número de melhores sensores a gerar (quando --auto-best).')
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    files = {
        '3': 'dataset_balanced_w_10_dx_3.csv',
        '4': 'dataset_balanced_w_10_dx_4.csv',
        '5': 'dataset_balanced_w_10_dx_5.csv',
    }

    per_dx_figs: List[str] = []
    summary_rows: List[Dict[str, float]] = []
    loaded: Dict[str, pd.DataFrame] = {}

    for dx, path in files.items():
        if not os.path.exists(path):
            print(f"Arquivo não encontrado: {path}")
            continue
        df = pd.read_csv(path)
        # Filtro por usuário, se solicitado
        if args.user is not None and 'user' in df.columns:
            df = df[df['user'] == args.user].copy()
        loaded[dx] = df

    # Seleção automática do melhor sensor (se solicitado ou se --sensor não vier)
    sensor = args.sensor
    if sensor is None or args.auto_best:
        candidates = _list_candidate_sensors(loaded)
        # Filtro por relevância para TMD (evitar sensores pouco informativos como pressure/proximity/step_counter)
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
        candidates = [c for c in candidates if c in relevance_allow]
        ranking: List[Tuple[str, float, float, int]] = []  # (sensor, delta_acc, delta_sil, n)
        for s in candidates:
            d_acc, d_sil, n = _score_sensor_across_dx(loaded, s, min_rows=args.min_rows)
            ranking.append((s, d_acc, d_sil, n))
        # ordenar por delta_acc (desc), desempate por delta_sil
        ranking.sort(key=lambda t: (t[1], t[2]), reverse=True)
        # Salvar ranking
        import csv
        rank_path = os.path.join(outdir, 'sensor_improvement_ranking.csv')
        with open(rank_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sensor', 'mean_delta_lr_acc', 'mean_delta_silhouette', 'num_dx'])
            for s, da, ds, n in ranking:
                writer.writerow([s, f"{da:.4f}", f"{ds:.4f}", n])
        print(f"Ranking salvo em: {rank_path}")
        if ranking:
            top_k = max(1, args.top_k)
            top_sensors = [r[0] for r in ranking[:top_k]]
            print(f"Sensores selecionados automaticamente (top {top_k}): {', '.join(top_sensors)}")
        else:
            raise SystemExit("Não foi possível identificar sensores válidos.")

        # Gerar saídas para os sensores escolhidos
        for sensor in top_sensors:
            per_dx_figs.clear()
            summary_rows.clear()
            for dx, df in loaded.items():
                if len(df) < args.min_rows:
                    print(f"{sensor} dx={dx}: menos de {args.min_rows} linhas após filtro; pulando.")
                    continue
                try:
                    X_base, X_it, y = select_sensor_feature_sets(df, sensor)
                except ValueError as e:
                    print(f"{sensor} dx={dx}: {e}")
                    continue
                if len(np.unique(y)) < 2:
                    print(f"{sensor} dx={dx}: menos de 2 classes encontradas; pulando.")
                    continue
                Z_base = project_pca(X_base)
                Z_it = project_pca(X_it)
                sil_base = silhouette_score(Z_base, y)
                sil_it = silhouette_score(Z_it, y)
                acc_base = evaluate_lr_acc(X_base, y)
                acc_it = evaluate_lr_acc(X_it, y)
                out_path = os.path.join(outdir, f"{safe_name(sensor)}_separability_pca_dx{dx}.png")
                plot_pca_side_by_side(Z_base, Z_it, y, sil_base, sil_it, acc_base, acc_it, sensor, dx, out_path)
                per_dx_figs.append(out_path)
                summary_rows.append({
                    'sensor': sensor,
                    'dx': dx,
                    'silhouette_baseline': sil_base,
                    'silhouette_with_it': sil_it,
                    'lr_acc_baseline': acc_base,
                    'lr_acc_with_it': acc_it,
                })
                print(
                    f"{sensor} dx={dx}: Silhouette (base->IT): {sil_base:.3f}->{sil_it:.3f} | "
                    f"LR-ACC (base->IT): {acc_base*100:.2f}%->{acc_it*100:.2f}%"
                )
            if summary_rows:
                df_sum = pd.DataFrame(summary_rows).sort_values('dx')
                csv_path = os.path.join(outdir, f"{safe_name(sensor)}_separability_summary.csv")
                df_sum.to_csv(csv_path, index=False)
                print(f"Resumo salvo em: {csv_path}")
            hc_path = os.path.join(outdir, f"{safe_name(sensor)}_entropy_complexity_dx3_4_5_grid.png")
            plot_hc_grid(loaded, sensor, hc_path)
            print(f"Plano H×C salvo em: {hc_path}")
    else:
        # Caminho para um único sensor especificado
        for dx, df in loaded.items():
            if len(df) < args.min_rows:
                print(f"{sensor} dx={dx}: menos de {args.min_rows} linhas após filtro; pulando.")
                continue
            try:
                X_base, X_it, y = select_sensor_feature_sets(df, sensor)
            except ValueError as e:
                print(f"{sensor} dx={dx}: {e}")
                continue
            if len(np.unique(y)) < 2:
                print(f"{sensor} dx={dx}: menos de 2 classes encontradas; pulando.")
                continue
            Z_base = project_pca(X_base)
            Z_it = project_pca(X_it)
            sil_base = silhouette_score(Z_base, y)
            sil_it = silhouette_score(Z_it, y)
            acc_base = evaluate_lr_acc(X_base, y)
            acc_it = evaluate_lr_acc(X_it, y)
            out_path = os.path.join(outdir, f"{safe_name(sensor)}_separability_pca_dx{dx}.png")
            plot_pca_side_by_side(Z_base, Z_it, y, sil_base, sil_it, acc_base, acc_it, sensor, dx, out_path)
            per_dx_figs.append(out_path)
            summary_rows.append({
                'sensor': sensor,
                'dx': dx,
                'silhouette_baseline': sil_base,
                'silhouette_with_it': sil_it,
                'lr_acc_baseline': acc_base,
                'lr_acc_with_it': acc_it,
            })
            print(
                f"{sensor} dx={dx}: Silhouette (base->IT): {sil_base:.3f}->{sil_it:.3f} | "
                f"LR-ACC (base->IT): {acc_base*100:.2f}%->{acc_it*100:.2f}%"
            )
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows).sort_values('dx')
            csv_path = os.path.join(outdir, f"{safe_name(sensor)}_separability_summary.csv")
            df_sum.to_csv(csv_path, index=False)
            print(f"Resumo salvo em: {csv_path}")
        hc_path = os.path.join(outdir, f"{safe_name(sensor)}_entropy_complexity_dx3_4_5_grid.png")
        plot_hc_grid(loaded, sensor, hc_path)
        print(f"Plano H×C salvo em: {hc_path}")


if __name__ == '__main__':
    main()


