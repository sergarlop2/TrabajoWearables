import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

activities = ["baseball", "bolos", "boxeo", "golf", "tenis", "reposo"]

# Load and preprocess all data
def load_all_data():
    all_data = []
    for activity in activities:
        df = pd.read_csv(f"datasets/datos_wii_{activity}.csv")
        df["actividad"] = activities.index(activity)  # Convert activity to numeric
        df["actividad"] = activities.index(activity)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Load the data
datos = load_all_data()

# Visualize activity distribution
plt.figure(figsize=(10, 5))
plt.hist(datos["actividad"], bins=len(activities), rwidth=0.8)
plt.xticks(range(len(activities)), activities, rotation=45)
plt.title("Activity Distribution")
plt.grid(True)
plt.show()