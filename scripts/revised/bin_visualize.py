import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PATIENTS_CSV_PATH = r"C:\Users\picul\Videos\Applied\Dataset_Full\patients.csv"
df = pd.read_csv(PATIENTS_CSV_PATH)

ahi_col = 'AHI'

if ahi_col not in df.columns:
    print(f"Columns available: {df.columns.tolist()}")
    ahi_col = df.columns[df.columns.str.lower().str.contains('ahi')][0]

df[ahi_col] = pd.to_numeric(df[ahi_col], errors='coerce')
df[ahi_col] = df[ahi_col].fillna(0.0)

asymptomatic = df[df[ahi_col] < 5]
mild = df[(df[ahi_col] >= 5) & (df[ahi_col] < 15)]
moderate = df[(df[ahi_col] >= 15) & (df[ahi_col] < 30)]
severe = df[df[ahi_col] >= 30]

counts = {
    'Asymptomatic\n(< 5)': len(asymptomatic),
    'Mild\n(5-14)': len(mild),
    'Moderate\n(15-29)': len(moderate),
    'Severe\n(≥ 30)': len(severe)
}

total = sum(counts.values())
percentages = {k: (v / total) * 100 for k, v in counts.items()}

print("AHI Severity Distribution:")
print("-" * 50)
for severity, count in counts.items():
    pct = percentages[severity]
    print(f"{severity.replace(chr(10), ' '):<20} | Count: {count:>3} | {pct:>6.2f}%")

print("-" * 50)
print(f"{'Total':<20} | Count: {total:>3} | 100.00%\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
bars = ax1.bar(counts.keys(), counts.values(), color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax1.set_xlabel('AHI Severity Category', fontsize=12, fontweight='bold')
ax1.set_title('AHI Distribution - Sample Counts', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, counts.values()):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(count)}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

wedges, texts, autotexts = ax2.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%',
                                     colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('AHI Distribution - Percentage', fontsize=13, fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('ahi_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

class_imbalance_ratio = counts['Severe\n(≥ 30)'] / counts['Asymptomatic\n(< 5)'] if counts['Asymptomatic\n(< 5)'] > 0 else float('inf')
print(f"Class Imbalance Ratio (Severe/Asymptomatic): {class_imbalance_ratio:.2f}x")