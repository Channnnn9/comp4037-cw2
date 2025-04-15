
# COMP4037 Research Methods CW2
# Chan wu
# 2025.4
# Tool: Python (matplotlib, pandas, sklearn)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:/Users/31634/Desktop/Results_21Mar2022.csv")
#df = pd.read_csv("Results_21Mar2022.csv")


selected_columns = [
    "diet_group",
    "mean_ghgs",
    "mean_land",
    "mean_watscar",
    "mean_eut",
    "mean_bio",
    "mean_watuse",
    "mean_acid"
]

# Group by diet_group , calculate means
df_grouped = df[selected_columns].groupby("diet_group").mean().reset_index()

# Sort by diet order
diet_order = ["vegan", "vegetarian", "fish", "meat"]
df_grouped["diet_group"] = pd.Categorical(df_grouped["diet_group"], categories=diet_order, ordered=True)
df_grouped = df_grouped.sort_values("diet_group")

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df_grouped[selected_columns[1:]])
data_normalized = np.concatenate([data_normalized, data_normalized[:, [0]]], axis=1)

labels = selected_columns[1:]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, row in enumerate(data_normalized):
    label = df_grouped["diet_group"].iloc[i]
    ax.plot(angles, row, label=label, linewidth=2)
    ax.fill(angles, row, alpha=0.1)

# Axis and title
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# Add legend and title
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.title("Environmental Impact by Diet Group (Normalized)", size=16)

plt.tight_layout()
plt.savefig("radar_chart_diet.png")
plt.show()
