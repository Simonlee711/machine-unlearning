import matplotlib.pyplot as plt
import random
# Plot setup
df = pd.read_csv("./unlearning_results_10.csv")
# Ensure necessary columns are calculated
df['AUC Diff'] = df['Unlearned AUC'] - df['Retrained AUC']
df['Mean AUC'] = (df['Unlearned AUC'] + df['Retrained AUC']) / 2

df['F1 Diff'] = df['Unlearned F1'] - df['Retrained F1']  # Replace with actual F1 columns if available
df['Mean F1'] = (df['Unlearned F1'] + df['Retrained F1']) / 2  # Replace with actual F1 columns if available

df['Log Loss Diff'] = df['Unlearned Log Loss'] - df['Retrained Log Loss']  # Replace with actual F1 columns if available
df['Mean Log Loss'] = (df['Unlearned Log Loss'] + df['Retrained Log Loss']) / 2  # Replace with actual F1 columns if available

# Metrics to plot
metrics = ['AUC', 'F1', 'Log Loss']
means = ['Mean AUC', 'Mean F1', "Mean Log Loss"]
differences = ['AUC Diff', 'F1 Diff', "Log Loss Diff"]

# Subplots for Bland-Altman plots
fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

for ax, metric, mean_col, diff_col in zip(axes, metrics, means, differences):
    # Calculate mean difference and standard deviation
    mean_diff = df[diff_col].mean()
    sd_diff = df[diff_col].std()
    
    # Bland-Altman scatter plot
    ax.scatter(df[mean_col], df[diff_col], alpha=0.7, label='Datasets', color='blue')
    ax.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Difference: {mean_diff:.3f}')
    ax.axhline(0, color='black', linestyle='--', label=f'zero')
    ax.axhline(mean_diff + 1.96 * sd_diff, color='red', linestyle='--', label=f'+1.96 SD: {mean_diff + 1.96 * sd_diff:.3f}')
    ax.axhline(mean_diff - 1.96 * sd_diff, color='red', linestyle='--', label=f'-1.96 SD: {mean_diff - 1.96 * sd_diff:.3f}')
    
    # Formatting
    ax.set_title(f"Bland-Altman Plot for {metric}", fontsize=14)
    ax.set_xlabel(f"Mean {metric} (Unlearned & Retrained)")
    ax.set_ylabel(f"Difference (Unlearned - Retrained)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

# Overall layout
fig.suptitle("Bland-Altman Plots for AUC, F1, and Log Loss Metrics", fontsize=16)
plt.tight_layout()
plt.show()
