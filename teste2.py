import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

def illustrate_entropy_complexity():
    np.random.seed(42)
    time_series = np.sin(np.linspace(0, 3*np.pi, 40)) + np.random.normal(0, 0.2, 40)
    
    start_idx = 10
    time_series[start_idx+3] = abs(time_series[start_idx+3]) + 0.2
    
    order = 4
    delay = 1
    
    plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, height_ratios=[1, 1.2], hspace=0.4, wspace=0.3)
    
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(time_series, 'b-', linewidth=2)
    ax1.set_title("(a) Time Series", fontsize=18)
    ax1.set_xlabel("Time", fontsize=18)
    ax1.set_ylabel("Amplitude", fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    indices = np.arange(start_idx, start_idx + order)
    ax1.plot(indices, time_series[indices], 'ro-', linewidth=3)
    ax1.text(start_idx + order/2 + 1, max(time_series[indices]) + 0.3, "Analysis Window", 
             ha='center', fontsize=16, color='red', fontweight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))  
    
    ax2 = plt.subplot(gs[1, 0])
    window = time_series[indices]
    
    bars = ax2.bar(np.arange(order), window, color='skyblue')
    ax2.set_title("(b) Values in Window", fontsize=18)
    ax2.set_xlabel("Position", fontsize=18)
    ax2.set_ylabel("Value", fontsize=18)
    ax2.set_xticks(np.arange(order))
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    y_min = min(window) - 0.3
    y_max = max(window) + 0.3
    ax2.set_ylim(y_min, y_max)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height >= 0:
            y_pos = height + 0.05
        else:
            y_pos = height - 0.15
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f'{window[i]:.2f}', ha='center', va='center', fontsize=14)
    
    ax3 = plt.subplot(gs[1, 1])
    sorted_indices = np.argsort(window)
    patterns = np.zeros(order)
    for i, idx in enumerate(sorted_indices):
        patterns[idx] = i
        
    bars = ax3.bar(np.arange(order), patterns + 1, color='lightgreen')
    ax3.set_title("(c) Ordinal Pattern (π)", fontsize=18)
    ax3.set_xlabel("Original Position", fontsize=18)
    ax3.set_ylabel("Rank", fontsize=18)
    ax3.set_xticks(np.arange(order))
    ax3.set_yticks(np.arange(1, order+1))
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_ylim(0.5, order + 1.5)
    
    for i, bar in enumerate(bars):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 f'{int(patterns[i] + 1)}', ha='center', fontsize=14)
        
    ax4 = plt.subplot(gs[1, 2])
    
    perms = list(itertools.permutations(range(order)))
    counts = {perm: 0 for perm in perms}
    
    for i in range(len(time_series) - order + 1):
        window = time_series[i:i + order]
        sorted_order = tuple(np.argsort(window))
        counts[sorted_order] += 1
    
    sorted_counts = sorted([(k, v) for k, v in counts.items()], key=lambda x: x[1], reverse=True)
    top_patterns = sorted_counts[:4]
    
    labels = [f"{i+1}" for i in range(len(top_patterns))]
    values = [v[1] for v in top_patterns]
    bars = ax4.bar(labels, values, color='salmon')
    ax4.set_title("(d) Pattern Distribution", fontsize=18)
    ax4.set_xlabel("Patterns", fontsize=18)
    ax4.set_ylabel("Frequency", fontsize=18)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    
    ax4.set_ylim(0, max(values) * 1.15)
    
    for i, bar in enumerate(bars):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 f'{values[i]}', ha='center', fontsize=14)
    
    ax5 = plt.subplot(gs[1, 3])
    
    total = float(sum(counts.values()))
    p = np.array([float(counts[perm]) / total for perm in perms])
    pe = -sum([pi * np.log(pi) if pi > 0 else 0 for pi in p])
    pe_norm = pe / np.log(math.factorial(order))
    
    M = len(p)
    u = np.ones(M) / M
    m = 0.5 * (p + u)
    
    def kl_divergence(a, b):
        return sum([a[i] * np.log(a[i] / b[i]) for i in range(len(a)) if a[i] > 0])
    
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(u, m)
    Q0 = -0.5 * (((M + 1.0) / M) * np.log((M + 1.0) / M) - 2 * np.log(2) + np.log(4.0 / M))
    QJ = js / Q0 if Q0 != 0 else 0
    complexity = QJ * pe_norm
    
    ax5.set_title("(e) Features", fontsize=18)
    ax5.set_xlim(-3.0, 4.0)
    ax5.set_ylim(-2.0, 2.0)
    ax5.axis('off')

    square_width = 3.0
    square_height = 2.5

    cmap_entropy = plt.cm.Blues
    cmap_complexity = plt.cm.Greens

    entropy_square = plt.Rectangle((-2.5, -1.25), square_width, square_height, 
                                   facecolor=cmap_entropy(0.6), edgecolor='black', linewidth=2)
    ax5.add_patch(entropy_square)
    ax5.text(-1.0, 0, f'H = {pe_norm:.3f}', ha='center', va='center', 
             fontsize=16, fontweight='bold')
    ax5.text(-1.0, -0.5, 'Entropy', ha='center', va='center', fontsize=16)

    complexity_square = plt.Rectangle((1.0, -1.25), square_width, square_height, 
                                      facecolor=cmap_complexity(0.6), edgecolor='black', linewidth=2)
    ax5.add_patch(complexity_square)
    ax5.text(2.5, 0, f'C = {complexity:.3f}', ha='center', va='center', 
             fontsize=16, fontweight='bold')
    ax5.text(2.5, -0.5, 'Complexity', ha='center', va='center', fontsize=16)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('entropy_complexity_illustration.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    illustrate_entropy_complexity()