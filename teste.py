import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects

# Atualize os parâmetros globais para que todo texto seja bold:
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 20,
    'font.weight': 'bold',          # Todo texto em bold
    'axes.titlesize': 22,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
})

# Paleta de cores padronizada
COLOR_WINDOW_ONLY = '#729ECE'  # Azul claro
COLOR_WINDOW_IT = '#FF9E4A'    # Laranja claro
COLOR_IMPROVEMENT = '#2ca02c'  # Verde

# Dados unificados
architectures = ['Traditional MLP', 'Federated MLP', 'Federated Ensemble']

# Organizando os dados em arrays para cada métrica
# [Traditional MLP, Federated MLP, Federated Ensemble]

# Window-10 Only
w10_only_acc = [90.92, 85.69, 89.54]
w10_only_prec = [92.41, 86.00, 90.00]  # Aproximado para Fed. Ensemble
w10_only_rec = [89.85, 85.00, 90.00]   # Aproximado para Fed. Ensemble
w10_only_f1 = [91.11, 86.00, 90.00]    # Aproximado para Fed. Ensemble

# Window-10 + IT Features
w10_it_acc = [94.00, 90.00, 95.08]
w10_it_prec = [94.12, 90.00, 95.00]    # Aproximado para Fed. Ensemble
w10_it_rec = [93.54, 90.00, 95.00]     # Aproximado para Fed. Ensemble
w10_it_f1 = [93.83, 90.00, 95.00]      # Aproximado para Fed. Ensemble

# Diferenças para cada métrica
diff_acc = [w10_it_acc[i] - w10_only_acc[i] for i in range(len(architectures))]
diff_prec = [w10_it_prec[i] - w10_only_prec[i] for i in range(len(architectures))]
diff_rec = [w10_it_rec[i] - w10_only_rec[i] for i in range(len(architectures))]
diff_f1 = [w10_it_f1[i] - w10_only_f1[i] for i in range(len(architectures))]

# Modificação na criação da figura e no espaçamento horizontal:
fig, axes = plt.subplots(2, 2, figsize=(26, 18))  # Aumenta a largura da figura para 20
axes = axes.flatten()  # Simplifica o acesso aos eixos

# Valores dx para anotações
dx_values = [5, 4, 3]

# Função para criar barras e anotações
def create_metric_chart(ax, window_only, window_it, differences, title):
    bar_width = 0.35
    x = np.arange(len(architectures))
    
    # Adicionar título ao gráfico
    ax.set_title(title, fontweight='bold', fontsize=26, pad=10)
    
    # Criação das barras
    bars1 = ax.bar(x - bar_width/2, window_only, bar_width, label='Baseline', 
                   color=COLOR_WINDOW_ONLY, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + bar_width/2, window_it, bar_width, label='Information theory', 
                   color=COLOR_WINDOW_IT, edgecolor='black', linewidth=1)
    
    # Adicionar indicação de melhoria (os textos já possuem fontweight='bold')
    for i in range(len(architectures)):
        if differences[i] > 0:
            txt = ax.annotate(
                f'+{differences[i]:.2f}%', 
                xy=(i, max(window_only[i], window_it[i]) + 1),
                xytext=(i, max(window_only[i], window_it[i]) + 3),
                ha='center', va='bottom', 
                fontsize=18,
                color=COLOR_IMPROVEMENT, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec=COLOR_IMPROVEMENT)
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Adicionar valores nas barras (todos os textos em bold)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{min(height, 100):.1f}%',  # Mostra valor máximo de 100%
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=24, fontweight='bold'
            )
    
    # Configurar layout com textos em bold nos rótulos
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, rotation=0, ha='center')
    plt.setp(ax.get_xticklabels(), fontsize=22, fontweight='bold')
    ax.set_ylim(70, 104)  # Limite vertical conforme solicitado
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, prop={'weight': 'bold'})
    
    # Legendas adicionais apenas no primeiro gráfico
    if title == 'Classification Accuracy':  # Modificado para o título em inglês
        for i in range(len(architectures)):
            ax.annotate(
                f'dx={dx_values[i]}', 
                xy=(i + bar_width/2, window_it[i]/2),
                ha='center', va='center',
                fontsize=20, color='white', fontweight='bold'
            )

# Criar os quatro gráficos com títulos em inglês
create_metric_chart(axes[0], w10_only_acc, w10_it_acc, diff_acc, 'Classification Accuracy')
create_metric_chart(axes[1], w10_only_prec, w10_it_prec, diff_prec, 'Precision (True Positives)')
create_metric_chart(axes[2], w10_only_rec, w10_it_rec, diff_rec, 'Recall (Class Coverage)')
create_metric_chart(axes[3], w10_only_f1, w10_it_f1, diff_f1, 'F1 Score (Harmonic Mean)')

# Título global e ajustes de espaçamento
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.5, hspace=0.3)  # wspace aumentado para 0.5

# Adicionar rótulos y nos eixos externos apenas
axes[0].set_ylabel('Accuracy (%)', fontweight='bold', fontsize=24)
axes[1].set_ylabel('Precision (%)', fontweight='bold', fontsize=24)
axes[2].set_ylabel('Recall (%)', fontweight='bold', fontsize=24)
axes[3].set_ylabel('F1 Score (%)', fontweight='bold', fontsize=24)

# Salvar o gráfico
plt.savefig('ablation_study_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()