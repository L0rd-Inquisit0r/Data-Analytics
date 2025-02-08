import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\jrudu\Documents\Code\Data Anal\Assignment_2_Data_Analytics\bar\bar_assignment.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
df.head()

# Convert 1 to "Yes" and 0 to "No"
df["COUNT"] = df["COUNT"].map({1: "Yes", 0: "No"})

# Count occurrences of each LABEL grouped by Yes/No
bar_data = df.groupby(["LABEL", "COUNT"]).size().unstack(fill_value=0)

# Plot horizontal bar chart
ax = bar_data.plot(kind="barh", stacked=True, figsize=(10, 6), color=["red", "blue"])

# Add labels to bars
for bars in ax.containers:
    ax.bar_label(bars, fmt='%d', label_type='edge', color='white', fontsize=12, fontweight='bold')

# Set title and axis labels
plt.title("TITLE OF PLOT HERE", fontsize=14, fontweight="bold")
plt.xlabel("X-LABELS", fontsize=12, fontweight="bold")
plt.ylabel("Y-LABELS", fontsize=12, fontweight="bold")

# Customize legend
plt.legend(title="LEGEND HERE", loc="upper left", fontsize=12, title_fontsize=12)

# Show the chart
plt.show()