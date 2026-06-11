import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

PATIENTS_CSV_PATH = (
    r"C:\Users\picul\Videos\Applied\Dataset_Full\patients.csv"
)
df = pd.read_csv(PATIENTS_CSV_PATH)

numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.drop(columns=["user_id", "night_id"])

numeric_df = numeric_df.fillna(0.0)
numeric_df = numeric_df.replace([np.inf, -np.inf], 0.0)

correlation_matrix = numeric_df.corr()

print("Correlation Matrix:")
print(correlation_matrix)
print("\n")

vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_df.columns

vif_values = []
for i in range(numeric_df.shape[1]):
    try:
        vif = variance_inflation_factor(numeric_df.values, i)
        vif_values.append(vif)
    except:
        vif_values.append(np.nan)

vif_data["VIF"] = vif_values

print("Variance Inflation Factor (VIF):")
print(vif_data)
print("\n")

high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

if high_corr_pairs:
    print("Highly Correlated Pairs (|r| > 0.7):")
    for pair in high_corr_pairs:
        print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
else:
    print("No highly correlated pairs found (|r| > 0.7)")

print("\n")
moderate_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if 0.5 < abs(correlation_matrix.iloc[i, j]) <= 0.7:
            moderate_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

if moderate_corr_pairs:
    print("Moderately Correlated Pairs (0.5 < |r| <= 0.7):")
    for pair in moderate_corr_pairs:
        print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f',
            annot_kws={'size': 9}, vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.set_palette("husl")

pairplot = sns.pairplot(
    numeric_df,
    diag_kind='kde',
    plot_kws={
        'alpha': 0.65,
        's': 50,
        'edgecolor': 'k',
        'linewidth': 0.5
    },
    diag_kws={
        'linewidth': 2,
        'shade': True
    },
    corner=False,
    height=2.5,
    aspect=1.2
)

pairplot.fig.suptitle('Feature Scatterplot Matrix with Distribution',
                      fontsize=16, fontweight='bold', y=0.995)

for ax in pairplot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatterplot_matrix.png', dpi=300, bbox_inches='tight')
plt.show()