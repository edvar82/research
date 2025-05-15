import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

def illustrate_entropy_complexity():
    # Create a simplified time series example
    np.random.seed(42)
    # Reduced size time series for space efficiency
    time_series = np.sin(np.linspace(0, 3*np.pi, 40)) + np.random.normal(0, 0.2, 40)
    
    # Force the 4th value in our analysis window to be positive
    start_idx = 10
    time_series[start_idx+3] = abs(time_series[start_idx+3]) + 0.2  # Ensure positive with small offset
    
    # Parameters
    order = 4
    delay = 1
    
    # Create figure
    plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, height_ratios=[1, 1.2], hspace=0.4, wspace=0.3)
    
    # 1. Display time series
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(time_series, 'b-', linewidth=2)
    ax1.set_title("(a) Time Series", fontsize=18)  # Panel label incorporated into title
    ax1.set_xlabel("Time", fontsize=18)
    ax1.set_ylabel("Amplitude", fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Highlight sample window
    indices = np.arange(start_idx, start_idx + order)
    ax1.plot(indices, time_series[indices], 'ro-', linewidth=3)
    ax1.text(start_idx + order/2 + 1, max(time_series[indices]) + 0.3, "Analysis Window", 
             ha='center', fontsize=16, color='red', fontweight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))  
    
    # 2. Demonstrate value pattern within the window
    ax2 = plt.subplot(gs[1, 0])
    window = time_series[indices]
    
    # Create bars for window values
    bars = ax2.bar(np.arange(order), window, color='skyblue')
    ax2.set_title("(b) Values in Window", fontsize=18)  # Panel label incorporated into title
    ax2.set_xlabel("Position", fontsize=18)
    ax2.set_ylabel("Value", fontsize=18)
    ax2.set_xticks(np.arange(order))
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Adjusting y-limits to better accommodate values
    y_min = min(window) - 0.3
    y_max = max(window) + 0.3
    ax2.set_ylim(y_min, y_max)
    
    # Add numerical values on bars with better positioning (without boxes)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height >= 0:
            y_pos = height + 0.05
        else:
            y_pos = height - 0.15
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f'{window[i]:.2f}', ha='center', va='center', fontsize=14)
    
    # 3. Show corresponding ordinal pattern - all positive
    ax3 = plt.subplot(gs[1, 1])
    sorted_indices = np.argsort(window)
    patterns = np.zeros(order)
    for i, idx in enumerate(sorted_indices):
        patterns[idx] = i
        
    # Create positive bars for ordinal pattern (ensuring all are positive)
    bars = ax3.bar(np.arange(order), patterns + 1, color='lightgreen')  # +1 to ensure all positive
    ax3.set_title("(c) Ordinal Pattern (π)", fontsize=18)  # Panel label incorporated into title
    ax3.set_xlabel("Original Position", fontsize=18)
    ax3.set_ylabel("Rank", fontsize=18)
    ax3.set_xticks(np.arange(order))
    ax3.set_yticks(np.arange(1, order+1))  # Adjusted for +1 offset
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_ylim(0.5, order + 1.5)  # Add space for annotations
    
    # Add values on bars (without boxes)
    for i, bar in enumerate(bars):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 f'{int(patterns[i] + 1)}', ha='center', fontsize=14)
    
    # Pattern text without box
    ax3.text(1.5, order + 0.9, f"Pattern: {tuple(sorted_indices)}", ha='center', fontsize=14)
    
    # 4. Histogram of patterns - limited to 4 patterns
    ax4 = plt.subplot(gs[1, 2])
    
    # Calculate pattern distribution
    perms = list(itertools.permutations(range(order)))
    counts = {perm: 0 for perm in perms}
    
    for i in range(len(time_series) - order + 1):
        window = time_series[i:i + order]
        sorted_order = tuple(np.argsort(window))
        counts[sorted_order] += 1
    
    # Show exactly 4 most frequent patterns
    sorted_counts = sorted([(k, v) for k, v in counts.items()], key=lambda x: x[1], reverse=True)
    top_patterns = sorted_counts[:4]
    
    # Create histogram
    labels = [f"{i+1}" for i in range(len(top_patterns))]
    values = [v[1] for v in top_patterns]
    bars = ax4.bar(labels, values, color='salmon')
    ax4.set_title("(d) Pattern Distribution", fontsize=18)  # Panel label incorporated into title
    ax4.set_xlabel("Patterns", fontsize=18)
    ax4.set_ylabel("Frequency", fontsize=18)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    
    # Set y-limit with extra space for annotations
    ax4.set_ylim(0, max(values) * 1.15)
    
    # Add values on top of bars (without boxes)
    for i, bar in enumerate(bars):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 f'{values[i]}', ha='center', fontsize=14)
    
    # 5. Visualization of Entropy and Complexity Features
    ax5 = plt.subplot(gs[1, 3])
    
    # Calculate entropy and complexity
    total = float(sum(counts.values()))
    p = np.array([float(counts[perm]) / total for perm in perms])
    pe = -sum([pi * np.log(pi) if pi > 0 else 0 for pi in p])
    pe_norm = pe / np.log(math.factorial(order))
    
    # Calculate complexity
    M = len(p)
    u = np.ones(M) / M
    m = 0.5 * (p + u)
    
    def kl_divergence(a, b):
        return sum([a[i] * np.log(a[i] / b[i]) for i in range(len(a)) if a[i] > 0])
    
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(u, m)
    Q0 = -0.5 * (((M + 1.0) / M) * np.log((M + 1.0) / M) - 2 * np.log(2) + np.log(4.0 / M))
    QJ = js / Q0 if Q0 != 0 else 0
    complexity = QJ * pe_norm
    
    # Create feature squares visualization
    # Expand the axis limits to accommodate larger squares
    ax5.set_title("(e) Features", fontsize=18)
    ax5.set_xlim(-3.0, 4.0)
    ax5.set_ylim(-2.0, 2.0)
    ax5.axis('off')  # Turn off axis

    # Increase the square dimensions
    square_width = 3.0  # Increased width
    square_height = 2.5  # Increased height

    # Create a custom colormap for the squares
    cmap_entropy = plt.cm.Blues
    cmap_complexity = plt.cm.Greens

    # Entropy square - repositioned for the larger square
    entropy_square = plt.Rectangle((-2.5, -1.25), square_width, square_height, 
                                   facecolor=cmap_entropy(0.6), edgecolor='black', linewidth=2)
    ax5.add_patch(entropy_square)
    ax5.text(-1.0, 0, f'H = {pe_norm:.3f}', ha='center', va='center', 
             fontsize=16, fontweight='bold')
    ax5.text(-1.0, -0.5, 'Entropy', ha='center', va='center', fontsize=16)

    # Complexity square - repositioned for the larger square
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