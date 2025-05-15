import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurar o estilo para parecer mais científico com fontes maiores
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18  # Aumentado de 14
plt.rcParams['axes.labelsize'] = 22  # Aumentado de 16
plt.rcParams['axes.titlesize'] = 24  # Aumentado de 18
plt.rcParams['xtick.labelsize'] = 18  # Aumentado de 14
plt.rcParams['ytick.labelsize'] = 18  # Aumentado de 14
plt.rcParams['legend.fontsize'] = 18  # Aumentado de 14
plt.rcParams['figure.titlesize'] = 26  # Aumentado de 20

# Criar diretório para salvar as imagens se não existir
os.makedirs('plan_images', exist_ok=True)

# Sensor específico
sensor = "android.sensor.orientation"

# Carregar datasets para diferentes valores de dx
try:
    dataset_dx3 = pd.read_csv('dataset_balanced_w_10_dx_3.csv')
    dataset_dx4 = pd.read_csv('dataset_balanced_w_10_dx_4.csv')
    dataset_dx5 = pd.read_csv('dataset_balanced_w_10_dx_5.csv')
    
    datasets = {
        "dx=3": dataset_dx3,
        "dx=4": dataset_dx4,
        "dx=5": dataset_dx5
    }
except FileNotFoundError as e:
    print(f"Erro ao carregar datasets: {e}")
    exit(1)

# Criar figura para o grid de plots com mais espaço para acomodar fontes maiores
fig, axes = plt.subplots(1, 3, figsize=(28, 10), dpi=300, sharey=True)
fig.suptitle(f'Entropy vs. Statistical Complexity for {sensor}', fontweight='bold', size=28, y=1.05)

# Criar plots para cada valor de dx
for i, (dx_label, dataset) in enumerate(datasets.items()):
    entropy_col = f"{sensor}#entropy"
    complexity_col = f"{sensor}#complexity"
    
    # Verificar se as colunas existem no dataset
    if entropy_col not in dataset.columns or complexity_col not in dataset.columns:
        print(f"Erro: Colunas para {sensor} não encontradas no dataset {dx_label}")
        continue
    
    # Criar o scatter plot com estilo científico
    ax = axes[i]
    scatter = sns.scatterplot(
        data=dataset,
        x=entropy_col,
        y=complexity_col,
        hue='target',
        palette='colorblind',
        s=100,  # Aumentado de 80
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5,
        ax=ax
    )
    
    # Configurar título e rótulos com estilo científico e fontes maiores
    ax.set_title(f'{dx_label}', fontweight='bold', pad=20, fontsize=24)
    ax.set_xlabel('Shannon Entropy', fontweight='bold', fontsize=22)
    
    # Adicionar rótulo do eixo y apenas para o primeiro gráfico
    if i == 0:
        ax.set_ylabel('Statistical Complexity', fontweight='bold', fontsize=22)
    else:
        ax.set_ylabel('')
    
    # Melhorar bordas do gráfico
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)  # Aumentado de 1.5
    ax.spines['bottom'].set_linewidth(2.0)  # Aumentado de 1.5
    
    # Remover a legenda de todos, exceto do último plot
    if i < 2:
        ax.get_legend().remove()

# Melhorar a legenda apenas para o último plot com fontes maiores
legend = axes[2].legend(title='Transport Mode', title_fontsize=22,  # Aumentado de 16
              bbox_to_anchor=(1.05, 0.5), loc='center left', frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Ajustar layout e espaçamento
plt.tight_layout()

# Criar nome de arquivo seguro
safe_filename = sensor.replace('.', '_').replace('/', '_')
filename = f"plan_images/{safe_filename}_dx3_4_5_grid_large_font.png"

# Salvar em alta resolução
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Gráfico grid para {sensor} (dx=3,4,5) com fontes maiores salvo como {filename}")